import base64
import collections
import copy
import io
import os
import re
import logging
import json
import hashlib
import numpy as np
import pandas as pd
import tempfile
import zipfile
from collections import Counter, namedtuple
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import date
from urllib.parse import unquote_plus, urlparse
from abc import ABC, abstractmethod
from typing import Dict, List
from enum import Enum
from io import BytesIO
from pathlib import PurePath
from PIL import Image, ImageOps
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.rdd import PipelinedRDD
from pyspark.sql import functions as sf, types, Column, Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import udf, pandas_udf, to_timestamp
from pyspark.sql.session import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    FloatType,
    FractionalType,
    IntegralType,
    LongType,
    StringType,
    TimestampType,
    StructType,
    ArrayType,
)
from pyspark.sql.utils import AnalysisException
from statsmodels.tsa.seasonal import STL


#  You may want to configure the Spark Context with the right credentials provider.
spark = SparkSession.builder.master("local").getOrCreate()
mode = None

JOIN_COLUMN_LIMIT = 10
DATAFRAME_AUTO_COALESCING_SIZE_THRESHOLD = 1073741824
ESCAPE_CHAR_PATTERN = re.compile("[{}]+".format(re.escape(".`")))
VALID_JOIN_TYPE = frozenset(
    [
        "anti",
        "cross",
        "full",
        "full_outer",
        "fullouter",
        "inner",
        "left",
        "left_anti",
        "left_outer",
        "left_semi",
        "leftanti",
        "leftouter",
        "leftsemi",
        "outer",
        "right",
        "right_outer",
        "rightouter",
        "semi",
    ],
)
DATE_SCALE_OFFSET_DESCRIPTION_SET = frozenset(["Business day", "Week", "Month", "Annual Quarter", "Year"])
DEFAULT_NODE_OUTPUT_KEY = "default"
OUTPUT_NAMES_KEY = "output_names"
SUPPORTED_TYPES = {
    BooleanType: "Boolean",
    FloatType: "Float",
    LongType: "Long",
    DoubleType: "Double",
    StringType: "String",
    DateType: "Date",
    TimestampType: "Timestamp",
}
JDBC_DEFAULT_NUMPARTITIONS = 2
DEFAULT_RANDOM_SEED = 838257247
PREPROCESS_TEMP_TABLE_NAME = "DataWrangerPushdownTempTable"


def capture_stdout(func, *args, **kwargs):
    """Capture standard output to a string buffer"""
    stdout_string = io.StringIO()
    with redirect_stdout(stdout_string):
        func(*args, **kwargs)
    return stdout_string.getvalue()


def convert_or_coerce(pandas_df, spark):
    """Convert pandas df to pyspark df and coerces the mixed cols to string"""
    try:
        return spark.createDataFrame(pandas_df)
    except TypeError as e:
        match = re.search(r".*field (\w+).*Can not merge type.*", str(e))
        if match is None:
            raise e
        mixed_col_name = match.group(1)
        # Coercing the col to string
        pandas_df[mixed_col_name] = pandas_df[mixed_col_name].astype("str")
        return pandas_df


def dedupe_columns(cols):
    """Dedupe and rename the column names after applying join operators. Rules:
        * First, ppend "_0", "_1" to dedupe and mark as renamed.
        * If the original df already takes the name, we will append more "_dup" as suffix til it's unique.
    """
    col_to_count = Counter(cols)
    duplicate_col_to_count = {col: col_to_count[col] for col in col_to_count if col_to_count[col] != 1}
    for i in range(len(cols)):
        col = cols[i]
        if col in duplicate_col_to_count:
            idx = col_to_count[col] - duplicate_col_to_count[col]
            new_col_name = f"{col}_{str(idx)}"
            while new_col_name in col_to_count:
                new_col_name += "_dup"
            cols[i] = new_col_name
            duplicate_col_to_count[col] -= 1
    return cols


def default_spark(value):
    return {DEFAULT_NODE_OUTPUT_KEY: value}


def default_spark_with_stdout(df, stdout):
    return {
        DEFAULT_NODE_OUTPUT_KEY: df,
        "stdout": stdout,
    }


def default_spark_with_trained_parameters(value, trained_parameters):
    return {DEFAULT_NODE_OUTPUT_KEY: value, "trained_parameters": trained_parameters}


def default_spark_with_trained_parameters_and_state(df, trained_parameters, state):
    return {DEFAULT_NODE_OUTPUT_KEY: df, "trained_parameters": trained_parameters, "state": state}


def dispatch(key_name, args, kwargs, funcs):
    """
    Dispatches to another operator based on a key in the passed parameters.
    This also slices out any parameters using the parameter_name passed in,
    and will reassemble the trained_parameters correctly after invocation.

    Args:
        key_name: name of the key in kwargs used to identify the function to use.
        args: dataframe that will be passed as the first set of parameters to the function.
        kwargs: keyword arguments that key_name will be found in; also where args will be passed to parameters.
                These are also expected to include trained_parameters if there are any.
        funcs: dictionary mapping from value of key_name to (function, parameter_name)
    """
    if key_name not in kwargs:
        raise OperatorCustomerError(f"Missing required parameter {key_name}")

    operator = kwargs[key_name]
    multi_column_operators = kwargs.get("multi_column_operators", [])

    if operator not in funcs:
        raise OperatorCustomerError(f"Invalid choice selected for {key_name}. {operator} is not supported.")

    func, parameter_name = funcs[operator]

    # Extract out the parameters that should be available.
    func_params = kwargs.get(parameter_name, {})
    if func_params is None:
        func_params = {}

    # Extract out any trained parameters.
    specific_trained_parameters = None
    if "trained_parameters" in kwargs:
        trained_parameters = kwargs["trained_parameters"]
        if trained_parameters is not None and parameter_name in trained_parameters:
            specific_trained_parameters = trained_parameters[parameter_name]
    func_params["trained_parameters"] = specific_trained_parameters

    result = spark_operator_with_escaped_column(
        func, args, func_params, multi_column_operators=multi_column_operators, operator_name=operator
    )

    # Check if the result contains any trained parameters and remap them to the proper structure.
    if result is not None and "trained_parameters" in result:
        existing_trained_parameters = kwargs.get("trained_parameters")
        updated_trained_parameters = result["trained_parameters"]

        if existing_trained_parameters is not None or updated_trained_parameters is not None:
            existing_trained_parameters = existing_trained_parameters if existing_trained_parameters is not None else {}
            existing_trained_parameters[parameter_name] = result["trained_parameters"]

            # Update the result trained_parameters so they are part of the original structure.
            result["trained_parameters"] = existing_trained_parameters
        else:
            # If the given trained parameters were None and the returned trained parameters were None, don't return
            # anything.
            del result["trained_parameters"]

    return result


def filter_timestamps_by_dates(df, timestamp_column, start_date=None, end_date=None):
    """Helper to filter dataframe by start and end date."""
    # ensure start date < end date, if both specified
    if start_date is not None and end_date is not None and pd.to_datetime(start_date) > pd.to_datetime(end_date):
        raise OperatorCustomerError(
            "Invalid combination of start and end date given. Start date should come before end date."
        )

    # filter by start date
    if start_date is not None:
        if pd.to_datetime(start_date) is pd.NaT:  # make sure start and end dates are datetime-castable
            raise OperatorCustomerError(
                f"Invalid start date given. Start date should be datetime-castable. Found: start date = {start_date}"
            )
        else:
            df = df.filter(
                sf.col(timestamp_column) >= sf.unix_timestamp(sf.lit(str(pd.to_datetime(start_date)))).cast("timestamp")
            )

    # filter by end date
    if end_date is not None:
        if pd.to_datetime(end_date) is pd.NaT:  # make sure start and end dates are datetime-castable
            raise OperatorCustomerError(
                f"Invalid end date given. Start date should be datetime-castable. Found: end date = {end_date}"
            )
        else:
            df = df.filter(
                sf.col(timestamp_column) <= sf.unix_timestamp(sf.lit(str(pd.to_datetime(end_date)))).cast("timestamp")
            )  # filter by start and end date

    return df


def format_sql_query_string(query_string):
    # Initial strip
    query_string = query_string.strip()

    # Remove semicolon.
    # This is for the case where this query will be wrapped by another query.
    query_string = query_string.rstrip(";")

    # Split lines and strip
    lines = query_string.splitlines()
    arr = []
    for line in lines:
        if not line.strip():
            continue
        line = line.strip()
        line = line.rstrip(";")
        arr.append(line)
    formatted_query_string = " ".join(arr)
    return formatted_query_string


def get_and_validate_join_keys(join_keys):
    join_keys_left = []
    join_keys_right = []
    for join_key in join_keys:
        left_key = join_key.get("left", "")
        right_key = join_key.get("right", "")
        if not left_key or not right_key:
            raise OperatorCustomerError("Missing join key: left('{}'), right('{}')".format(left_key, right_key))
        join_keys_left.append(left_key)
        join_keys_right.append(right_key)

    if len(join_keys_left) > JOIN_COLUMN_LIMIT:
        raise OperatorCustomerError("We only support join on maximum 10 columns for one operation.")
    return join_keys_left, join_keys_right


def get_dataframe_with_sequence_ids(df: DataFrame):
    df_cols = df.columns
    rdd_with_seq = df.rdd.zipWithIndex()
    df_with_seq = rdd_with_seq.toDF()
    df_with_seq = df_with_seq.withColumnRenamed("_2", "_seq_id_")
    for col_name in df_cols:
        df_with_seq = df_with_seq.withColumn(col_name, df_with_seq["_1"].getItem(col_name))
    df_with_seq = df_with_seq.drop("_1")
    return df_with_seq


def get_execution_state(status: str, message=None):
    return {"status": status, "message": message}


def get_trained_params_by_col(trained_params, col):
    if isinstance(trained_params, list):
        for params in trained_params:
            if params.get("input_column") == col:
                return params
        return None
    return trained_params


def multi_output_spark(outputs_dict, handle_default=True):
    if handle_default and DEFAULT_NODE_OUTPUT_KEY in outputs_dict.keys():
        # Ensure 'default' is first in the list of output names if it is used
        output_names = [DEFAULT_NODE_OUTPUT_KEY]
        output_names.extend([key for key in outputs_dict.keys() if key != DEFAULT_NODE_OUTPUT_KEY])
    else:
        output_names = [key for key in outputs_dict.keys()]
    outputs_dict[OUTPUT_NAMES_KEY] = output_names
    return outputs_dict


def rename_invalid_column(df, orig_col):
    """Rename a given column in a data frame to a new valid name

    Args:
        df: Spark dataframe
        orig_col: input column name

    Returns:
        a tuple of new dataframe with renamed column and new column name
    """
    temp_col = orig_col
    if ESCAPE_CHAR_PATTERN.search(orig_col):
        idx = 0
        temp_col = ESCAPE_CHAR_PATTERN.sub("_", orig_col)
        name_set = set(df.columns)
        while temp_col in name_set:
            temp_col = f"{temp_col}_{idx}"
            idx += 1
        df = df.withColumnRenamed(orig_col, temp_col)
    return df, temp_col


def spark_operator_with_escaped_column(
    operator_func,
    func_args,
    func_params,
    multi_column_operators=[],
    operator_name="",
    output_name=DEFAULT_NODE_OUTPUT_KEY,
):
    """Invoke operator func with input dataframe that has its column names sanitized.

    This function renames column names with special char to an internal name and
    rename it back after invocation

    Args:
        operator_func: underlying operator function
        func_args: operator function positional args, this only contains one element `df` for now
        func_params: operator function kwargs
        multi_column_operators: list of operators that support multiple columns, value of '*' indicates
        support all
        operator_name: operator name defined in node parameters
        output_name: the name of the output in the operator function result

    Returns:
        a dictionary with operator results
    """
    renamed_columns = {}
    input_column_key = multiple_input_column_key = "input_column"
    valid_output_column_keys = {"output_column", "output_prefix", "output_column_prefix"}
    is_output_col_key = set(func_params.keys()).intersection(valid_output_column_keys)
    output_column_key = list(is_output_col_key)[0] if is_output_col_key else None
    output_trained_params = []

    if input_column_key in func_params:
        # Copy on write so the original func_params is untouched to ensure inference mode correctness
        func_params = func_params.copy()
        # Convert input_columns to list if string ensuring backwards compatibility with strings
        input_columns = (
            func_params[input_column_key]
            if isinstance(func_params[input_column_key], list)
            else [func_params[input_column_key]]
        )

        # rename columns if needed
        sanitized_input_columns = []
        for input_col_value in input_columns:
            input_df, temp_col_name = rename_invalid_column(func_args[0], input_col_value)
            func_args[0] = input_df
            if temp_col_name != input_col_value:
                renamed_columns[input_col_value] = temp_col_name
            sanitized_input_columns.append(temp_col_name)

        iterate_over_multiple_columns = multiple_input_column_key in func_params and any(
            op_name in multi_column_operators for op_name in ["*", operator_name]
        )
        if not iterate_over_multiple_columns and len(input_columns) > 1:
            raise OperatorCustomerError(
                f"Operator {operator_name} does not support multiple columns, please provide a single column"
            )

        # output_column name as prefix if
        # 1. there are multiple input columns
        # 2. the output_column_key exists in params
        # 3. the output_column_value is not an empty string
        output_column_name = func_params.get(output_column_key)
        append_column_name_to_output_column = (
            iterate_over_multiple_columns and len(input_columns) > 1 and output_column_name
        )

        result = None
        trained_params_mul_cols = func_params.get("trained_parameters")

        # invalidate trained params if not type list for multi-column use case
        if len(sanitized_input_columns) > 1 and isinstance(trained_params_mul_cols, dict):
            trained_params_mul_cols = func_params["trained_parameters"] = None

        for input_col_val in sanitized_input_columns:
            if trained_params_mul_cols:
                func_params["trained_parameters"] = get_trained_params_by_col(trained_params_mul_cols, input_col_val)
            func_params[input_column_key] = input_col_val
            # if more than 1 column, output column name behaves as a prefix,
            if append_column_name_to_output_column:
                func_params[output_column_key] = f"{output_column_name}_{input_col_val}"

            # invoke underlying function on each column if multiple are present
            result = operator_func(*func_args, **func_params)
            func_args[0] = result[output_name]

            if result.get("trained_parameters"):
                # add input column to remove dependency on list order
                trained_params_copy = result["trained_parameters"].copy()
                trained_params_copy["input_column"] = input_col_val
                output_trained_params.append(trained_params_copy)
    else:
        # invoke underlying function
        result = operator_func(*func_args, **func_params)

    # put renamed columns back if applicable
    if result is not None and output_name in result:
        result_df = result[output_name]
        # rename col
        for orig_col_name, temp_col_name in renamed_columns.items():
            if temp_col_name in result_df.columns:
                result_df = result_df.withColumnRenamed(temp_col_name, orig_col_name)
        result[output_name] = result_df

    if len(output_trained_params) > 1:
        result["trained_parameters"] = output_trained_params

    return result


def stl_decomposition(ts, period=None):
    """Completes a Season-Trend Decomposition using LOESS (Cleveland et. al. 1990) on time series data.

    Parameters
    ----------
    ts: pandas.Series, index must be datetime64[ns] and values must be int or float.
    period: int, primary periodicity of the series. Default is None, will apply a default behavior
        Default behavior:
            if timestamp frequency is minute: period = 1440 / # of minutes between consecutive timestamps
            if timestamp frequency is second: period = 3600 / # of seconds between consecutive timestamps
            if timestamp frequency is ms, us, or ns: period = 1000 / # of ms/us/ns between consecutive timestamps
            else: defer to statsmodels' behavior, detailed here:
                https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/tsatools.py#L776

    Returns
    -------
    season: pandas.Series, index is same as ts, values are seasonality of ts
    trend: pandas.Series, index is same as ts, values are trend of ts
    resid: pandas.Series, index is same as ts, values are the remainder (original signal, subtract season and trend)
    """
    # TODO: replace this with another, more complex method for finding a better period
    period_sub_hour = {
        "T": 1440,  # minutes
        "S": 3600,  # seconds
        "M": 1000,  # milliseconds
        "U": 1000,  # microseconds
        "N": 1000,  # nanoseconds
    }
    if period is None:
        freq = ts.index.freq
        if freq is None:
            freq = pd.tseries.frequencies.to_offset(pd.infer_freq(ts.index))
        if freq is None:  # if still none, datetimes are not uniform, so raise error
            raise OperatorCustomerError(
                f"No uniform datetime frequency detected. Make sure the column contains datetimes that are evenly spaced (Are there any missing values?)"
            )
        for k, v in period_sub_hour.items():
            # if freq is not in period_sub_hour, then it is hourly or above and we don't have to set a default
            if k in freq.name:
                period = int(v / int(freq.n))  # n is always >= 1
                break
    model = STL(ts, period=period)
    decomposition = model.fit()
    return decomposition.seasonal, decomposition.trend, decomposition.resid, model.period


def to_timestamp_single(x):
    """Helper function for auto-detecting datetime format and casting to ISO-8601 string."""
    converted = pd.to_datetime(x, errors="coerce")
    return converted.astype("str").replace("NaT", "")  # makes pandas NaT into empty string


def to_vector(df, array_column):
    """Helper function to convert the array column in df to vector type column"""
    _udf = sf.udf(lambda r: Vectors.dense(r), VectorUDT())
    df = df.withColumn(array_column, _udf(array_column))
    return df


def uniform_sample(df, target_example_num, n_rows=None, min_required_rows=None):
    if n_rows is None:
        n_rows = df.count()
    if min_required_rows and n_rows < min_required_rows:
        raise OperatorCustomerError(
            f"Not enough valid rows available. Expected a minimum of {min_required_rows}, but the dataset contains "
            f"only {n_rows}"
        )
    sample_ratio = min(1, 3.0 * target_example_num / n_rows)
    return df.sample(withReplacement=False, fraction=float(sample_ratio), seed=0).limit(target_example_num)


def use_scientific_notation(values):
    """
    Return whether or not to use scientific notation in visualization's y-axis.

    Parameters
    ----------
    values: numpy array of values being plotted

    Returns
    -------
    boolean, True if viz should use scientific notation, False if not
    """
    _min = np.min(values)
    _max = np.max(values)
    _range = abs(_max - _min)
    return not (
        _range > 1e-3 and _range < 1e3 and abs(_min) > 1e-3 and abs(_min) < 1e3 and abs(_max) > 1e-3 and abs(_max) < 1e3
    )


def validate_col_name_in_df(col, df_cols):
    if col not in df_cols:
        raise OperatorCustomerError("Cannot resolve column name '{}'.".format(col))


def validate_join_type(join_type):
    if join_type not in VALID_JOIN_TYPE:
        raise OperatorCustomerError(
            "Unsupported join type '{}'. Supported join types include: {}.".format(
                join_type, ", ".join(VALID_JOIN_TYPE)
            )
        )


class OperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


import datetime


from pyspark.sql.types import (
    DoubleType,
    StringType,
    IntegralType,
    NumericType,
    IntegerType,
    ShortType,
    LongType,
    ByteType,
    FloatType,
    DecimalType,
    DateType,
    TimestampType,
    BooleanType,
)
import pyspark.sql.functions as sf

#  numeric types for handle missing operator
NUMERIC_DATATYPES = {IntegerType, ShortType, LongType, DoubleType, FloatType, ByteType, DecimalType}


def handle_missing_get_indicator_column(df, input_column, expected_type):
    """Helper function used to get an indicator for all missing values."""
    dcol = df[input_column].cast(expected_type)
    if isinstance(expected_type, StringType):
        indicator = sf.isnull(dcol) | (sf.trim(dcol) == "")
    elif isinstance(expected_type, (DateType, TimestampType)):
        indicator = sf.col(input_column).isNull()
    else:
        indicator = sf.isnull(dcol) | sf.isnan(dcol)
    return indicator


def handle_missing_replace_missing_values(df, input_column, output_column, impute_value, expected_type):
    """Helper function that replaces any missing values with the impute value."""

    expects_column(df, input_column, "Input column")

    if not isinstance(expected_type, (DoubleType, StringType, LongType, DateType, TimestampType)):
        raise OperatorSparkOperatorCustomerError(f"Data Wrangler does not support imputation for type {expected_type.typeName()}.")
    # Set output to default to input column if None or empty
    output_column = input_column if not output_column else output_column

    # Create a temp missing indicator column
    missing_col = temp_col_name(df)
    try:
        output_df = df.withColumn(missing_col, handle_missing_get_indicator_column(df, input_column, expected_type))
    except OverflowError as err:
        raise OperatorSparkOperatorCustomerError(f"Value in column {input_column} is out of range.")

    # Fill values and drop the temp indicator column

    output_df = output_df.withColumn(
        output_column,
        sf.when(output_df[missing_col] == 0, output_df[input_column]).otherwise(impute_value).cast(expected_type),
    ).drop(missing_col)

    return output_df


def handle_missing_numeric(df, input_column=None, output_column=None, strategy=None, trained_parameters=None):
    STRATEGY_MEAN = "Mean"
    STRATEGY_APPROXIMATE_MEDIAN = "Approximate Median"

    MEDIAN_RELATIVE_ERROR = 0.001

    # validate column name and type
    expects_column(df, input_column, "Input column")
    if not isinstance(df.schema[input_column].dataType, tuple(NUMERIC_DATATYPES)):
        raise OperatorSparkOperatorCustomerError(
            f"Data Wrangler can't calculate the imputation value for the column {input_column}. "
            f"Choose a numeric column or select 'Categorical' for 'Column type'."
        )

    trained_parameters = load_trained_parameters(trained_parameters, {"strategy": strategy})
    impute_value = parse_parameter(float, trained_parameters.get("impute_value"), "Trained parameters", nullable=True)
    if impute_value is None:
        if strategy == STRATEGY_MEAN:
            impute_value = (
                df.withColumn(input_column, df[input_column].cast(DoubleType()))
                .na.drop()
                .groupBy()
                .mean(input_column)
                .collect()[0][0]
            )
        elif strategy == STRATEGY_APPROXIMATE_MEDIAN:
            impute_value = df.withColumn(input_column, df[input_column].cast(DoubleType())).approxQuantile(
                input_column, [0.5], MEDIAN_RELATIVE_ERROR
            )[0]
        else:
            raise OperatorInternalError(
                f"Unexpected things happened. Invalid imputation strategy specified: {strategy}"
            )
        trained_parameters["impute_value"] = impute_value

    output_df = handle_missing_replace_missing_values(df, input_column, output_column, impute_value, DoubleType())

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def handle_missing_categorical(df, input_column=None, output_column=None, trained_parameters=None):
    # validate column  name and type
    expects_column(df, input_column, "Input column")
    expected_type = df.schema[input_column].dataType
    single_col = df.select(input_column).filter(~handle_missing_get_indicator_column(df, input_column, expected_type))
    trained_parameters = load_trained_parameters(trained_parameters, {})
    impute_value = parse_parameter(str, trained_parameters.get("impute_value"), "Trained parameters", nullable=True)
    date_type_mapping = {DateType: datetime.date, TimestampType: datetime.datetime}

    if impute_value and isinstance(expected_type, (TimestampType, DateType)):
        impute_value = date_type_mapping[expected_type.__class__].fromisoformat(impute_value)
    elif impute_value is None:
        try:
            top2counts = single_col.groupby(input_column).count().sort("count", ascending=False).head(2)
            impute_value = None
            for row in top2counts:
                if row[input_column] is not None:
                    impute_value = row[input_column]
                    break
            if isinstance(expected_type, (TimestampType, DateType)):
                trained_parameters["impute_value"] = date_type_mapping[expected_type.__class__].isoformat(impute_value)
            else:
                trained_parameters["impute_value"] = impute_value
        except Exception:
            raise OperatorSparkOperatorCustomerError(
                f"Could not calculate imputation value. Please ensure your column contains multiple values."
            )

    output_df = handle_missing_replace_missing_values(df, input_column, output_column, impute_value, expected_type)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def handle_missing_impute(df, **kwargs):
    kwargs["multi_column_operators"] = ["*"]
    return dispatch(
        "column_type",
        [df],
        kwargs,
        {
            "Numeric": (handle_missing_numeric, "numeric_parameters"),
            "Categorical": (handle_missing_categorical, "categorical_parameters"),
        },
    )


def handle_missing_fill_missing(df, input_column=None, output_column=None, fill_value=None, trained_parameters=None):
    expects_column(df, input_column, "Input column")
    if isinstance(df.schema[input_column].dataType, IntegralType):
        fill_value = parse_parameter(int, fill_value, "Fill Value")
    elif isinstance(df.schema[input_column].dataType, NumericType):
        fill_value = parse_parameter(float, fill_value, "Fill Value")

    output_df = handle_missing_replace_missing_values(
        df, input_column, output_column, fill_value, df.schema[input_column].dataType
    )

    return default_spark(output_df)


def handle_missing_add_indicator_for_missing(df, input_column=None, output_column=None, trained_parameters=None):
    expects_column(df, input_column, "Input column")
    indicator = handle_missing_get_indicator_column(df, input_column, df.schema[input_column].dataType)
    output_column = f"{input_column}_indicator" if not output_column else output_column
    df = df.withColumn(output_column, indicator)

    return default_spark(df)


def handle_missing_drop_rows(df, input_column=None, dimension=None, drop_rows_parameters=None, trained_parameters=None):
    """
    dimension and drop_rows_parameters are the old interface, we keep them from backward compatibility
    input_column is the new interface
    """
    if dimension:
        # old interface is used - convert to new interface
        assert dimension == "Drop Rows"
        input_column = drop_rows_parameters["input_column"]

    indicator_col_name = temp_col_name(df)
    if input_column:
        expects_column(df, input_column, "Input column")
        indicator = handle_missing_get_indicator_column(df, input_column, df.schema[input_column].dataType)
        output_df = df.withColumn(indicator_col_name, indicator)
    else:
        output_df = df
        for f in df.schema.fields:
            indicator = handle_missing_get_indicator_column(df, "`" + f.name + "`", f.dataType)
            if indicator_col_name in output_df.columns:
                output_df = output_df.withColumn(
                    indicator_col_name, sf.when(indicator | output_df[indicator_col_name], True).otherwise(False)
                )
            else:
                output_df = df.withColumn(indicator_col_name, indicator)
    output_df = output_df.where(f"{indicator_col_name} == 0").drop(indicator_col_name)
    return default_spark(output_df)




ImageData = namedtuple("ImageData", ["filename", "image"])

IMAGE_PREVIEW_WIDTH = 200
IMAGE_PREVIEW_HEIGHT = 200


def make_image_rdd(df: DataFrame, mode: str) -> PipelinedRDD:
    """
    Convert a 1-column df (origin) to RDD containing filenames and image objects.

    Parameters:
        df: pyspark DataFrame[origin: string]

    Returns:
        PipelinedRDD which evaluates to a list of tuples of the form (filename, image)
    """
    import boto3

    def fpath_to_pil(row):
        try:
            parse_result = urlparse(row.origin)
            bucket_name = parse_result.netloc
            prefix = parse_result.path.replace("/", "", 1)
            s3_client = boto3.client("s3")
            response = s3_client.get_object(Bucket=bucket_name, Key=prefix)["Body"].read()
            image = Image.open(BytesIO(response))
        except Exception as e:
            logging.warning(
                f"Image {prefix} could not be read. Verify access permissions and that the image is not corrupted: {e}"
            )
            image = None
        return ImageData(filename=PurePath(prefix).name, image=image)


    image_rdd = df.rdd.map(fpath_to_pil)

    return image_rdd


def grayscale_images(rdd, **kwargs):
    """
    Grayscales all images.

    Parameters:
        rdd: PythonRDD that evaulates to a list of PIL images.

    Returns:
        PythonRDD with grayscaled images.
    """

    def grayscale_func(item):
        if item.image is None:
            return ImageData(filename=item.filename, image=None)
        try:
            new_img = ImageOps.grayscale(item.image)
        except Exception as e:
            logging.error(f"Failed to grayscale image {item.filename}: {e}")
            new_img = None
        return ImageData(filename=item.filename, image=new_img)

    return default_spark(rdd.map(grayscale_func))




class NonCastableDataHandlingMethod(Enum):
    REPLACE_WITH_NULL = "replace_null"
    REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_null_with_new_col"
    REPLACE_WITH_FIXED_VALUE = "replace_value"
    REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_value_with_new_col"
    DROP_NON_CASTABLE_ROW = "drop"

    @staticmethod
    def get_names():
        return [item.name for item in NonCastableDataHandlingMethod]

    @staticmethod
    def get_values():
        return [item.value for item in NonCastableDataHandlingMethod]


class MohaveDataType(Enum):
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    FLOAT = "float"
    LONG = "long"
    STRING = "string"
    ARRAY = "array"
    STRUCT = "struct"
    OBJECT = "object"

    @staticmethod
    def get_names():
        return [item.name for item in MohaveDataType]

    @staticmethod
    def get_values():
        return [item.value for item in MohaveDataType]


PYTHON_TYPE_MAPPING = {
    MohaveDataType.BOOL: bool,
    MohaveDataType.DATE: str,
    MohaveDataType.DATETIME: str,
    MohaveDataType.FLOAT: float,
    MohaveDataType.LONG: int,
    MohaveDataType.STRING: str,
    MohaveDataType.ARRAY: str,
    MohaveDataType.STRUCT: str,
}

MOHAVE_TO_SPARK_TYPE_MAPPING = {
    MohaveDataType.BOOL: BooleanType,
    MohaveDataType.DATE: DateType,
    MohaveDataType.DATETIME: TimestampType,
    MohaveDataType.FLOAT: DoubleType,
    MohaveDataType.LONG: LongType,
    MohaveDataType.STRING: StringType,
    MohaveDataType.ARRAY: ArrayType,
    MohaveDataType.STRUCT: StructType,
}

SPARK_TYPE_MAPPING_TO_SQL_TYPE = {
    BooleanType: "BOOLEAN",
    LongType: "BIGINT",
    DoubleType: "DOUBLE",
    StringType: "STRING",
    DateType: "DATE",
    TimestampType: "TIMESTAMP",
}

SPARK_TO_MOHAVE_TYPE_MAPPING = {value: key for (key, value) in MOHAVE_TO_SPARK_TYPE_MAPPING.items()}


def cast_column_helper(df, column, mohave_data_type, date_col, datetime_col, non_date_col):
    """Helper for casting a single column to a data type."""
    if mohave_data_type == MohaveDataType.DATE:
        return df.withColumn(column, date_col)
    elif mohave_data_type == MohaveDataType.DATETIME:
        return df.withColumn(column, datetime_col)
    else:
        return df.withColumn(column, non_date_col)


def cast_single_column_type(
    df,
    column,
    mohave_data_type,
    invalid_data_handling_method,
    replace_value=None,
    date_formatting="dd-MM-yyyy",
    datetime_formatting=None,
):
    """Cast single column to a new type

    Args:
        df (DataFrame): spark dataframe
        column (Column): target column for type casting
        mohave_data_type (Enum): Enum MohaveDataType
        invalid_data_handling_method (Enum): Enum NonCastableDataHandlingMethod
        replace_value (str): value to replace for invalid data when "replace_value" is specified
        date_formatting (str): format for date. Default format is "dd-MM-yyyy"
        datetime_formatting (str): format for datetime. Default is None, indicates auto-detection

    Returns:
        df (DataFrame): casted spark dataframe
    """
    cast_to_date = sf.to_date(df[column], date_formatting)
    to_ts = sf.pandas_udf(f=to_timestamp_single, returnType="string")
    if datetime_formatting is None:
        cast_to_datetime = sf.to_timestamp(to_ts(df[column]))  # auto-detect formatting
    else:
        cast_to_datetime = sf.to_timestamp(df[column], datetime_formatting)
    cast_to_non_date = df[column].cast(MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type]())
    non_castable_column = f"{column}_typecast_error"
    temp_column = "temp_column"

    if invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_NULL:
        # Replace non-castable data to None in the same column. pyspark's default behaviour
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | None |
        # | 2 | None |
        # | 3 | 1    |
        # +---+------+
        return cast_column_helper(
            df,
            column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
    if invalid_data_handling_method == NonCastableDataHandlingMethod.DROP_NON_CASTABLE_ROW:
        # Drop non-castable row
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, _ non-castable row
        # +---+----+
        # | id|txt |
        # +---+----+
        # |  3|  1 |
        # +---+----+
        df = cast_column_helper(
            df,
            column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
        return df.where(df[column].isNotNull())

    if (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to None in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|None|      foo         |
        # |  2|None|      bar         |
        # |  3|  1 |                  |
        # +---+----+------------------+
        df = cast_column_helper(
            df,
            temp_column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
        df = df.withColumn(non_castable_column, sf.when(df[temp_column].isNotNull(), "").otherwise(df[column]),)
    elif invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE:
        # Replace non-castable data to a value in the same column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+-----+
        # | id| txt |
        # +---+-----+
        # |  1|  0  |
        # |  2|  0  |
        # |  3|  1  |
        # +---+----+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = cast_column_helper(
            df,
            temp_column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )

        replace_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            sf.to_date(sf.lit(value), date_formatting)
        )
        replace_non_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    elif (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to a value in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|  0  |   foo           |
        # |  2|  0  |   bar           |
        # |  3|  1  |                 |
        # +---+----+------------------+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = cast_column_helper(
            df,
            temp_column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
        df = df.withColumn(non_castable_column, sf.when(df[temp_column].isNotNull(), "").otherwise(df[column]),)

        replace_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            sf.to_date(sf.lit(value), date_formatting)
        )
        replace_non_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    # drop temporary column
    df = df.withColumn(column, df[temp_column]).drop(temp_column)

    df_cols = df.columns
    if non_castable_column in df_cols:
        # Arrange columns so that non_castable_column col is next to casted column
        df_cols.remove(non_castable_column)
        column_index = df_cols.index(column)
        arranged_cols = df_cols[: column_index + 1] + [non_castable_column] + df_cols[column_index + 1 :]
        df = df.select(*arranged_cols)
    return df


def _validate_and_cast_value(value, mohave_data_type):
    if value is None:
        return value
    try:
        return PYTHON_TYPE_MAPPING[mohave_data_type](value)
    except ValueError as e:
        raise ValueError(
            f"Invalid value to replace non-castable data. "
            f"{mohave_data_type} is not in mohave supported date type: {MohaveDataType.get_values()}. "
            f"Please use a supported type",
            e,
        )





class OperatorSparkOperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


def is_inference_running_mode():
    return False


def temp_col_name(df, *illegal_names, prefix: str = "temp_col"):
    """Generates a temporary column name that is unused.
    """
    name = prefix
    idx = 0
    name_set = set(list(df.columns) + list(illegal_names))
    while name in name_set:
        name = f"_{prefix}_{idx}"
        idx += 1

    return name


def get_temp_col_if_not_set(df, col_name):
    """Extracts the column name from the parameters if it exists, otherwise generates a temporary column name.
    """
    if col_name:
        return col_name, False
    else:
        return temp_col_name(df), True


def replace_input_if_output_is_temp(df, input_column, output_column, output_is_temp):
    """Replaces the input column in the dataframe if the output was not set

    This is used with get_temp_col_if_not_set to enable the behavior where a 
    transformer will replace its input column if an output is not specified.
    """
    if output_is_temp:
        df = df.withColumn(input_column, df[output_column])
        df = df.drop(output_column)
        return df
    else:
        return df


def parse_parameter(typ, value, key, default=None, nullable=False):
    if value is None:
        if default is not None or nullable:
            return default
        else:
            raise OperatorSparkOperatorCustomerError(f"Missing required input: '{key}'")
    else:
        try:
            value = typ(value)
            if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
                if np.isnan(value) or np.isinf(value):
                    raise OperatorSparkOperatorCustomerError(
                        f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
                    )
                else:
                    return value
            else:
                return value
        except (ValueError, TypeError):
            raise OperatorSparkOperatorCustomerError(
                f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
            )
        except OverflowError:
            raise OperatorSparkOperatorCustomerError(
                f"Overflow Error: Invalid value provided for '{key}'. Given value '{value}' exceeds the range of type "
                f"'{typ.__name__}' for this input. Insert a valid value for type '{typ.__name__}' and try your request "
                f"again."
            )


def expects_valid_column_name(value, key, nullable=False):
    if nullable and value is None:
        return

    if value is None or len(str(value).strip()) == 0:
        raise OperatorSparkOperatorCustomerError(f"Column name cannot be null, empty, or whitespace for parameter '{key}': {value}")


def expects_parameter(value, key, condition=None):
    if value is None:
        raise OperatorSparkOperatorCustomerError(f"Missing required input: '{key}'")
    elif condition is not None and not condition:
        raise OperatorSparkOperatorCustomerError(f"Invalid value provided for '{key}': {value}")


def expects_column(df, value, key):
    if not value or value not in df.columns:
        raise OperatorSparkOperatorCustomerError(
            f"The column '{value}' does not exist in your dataset. For '{key}', specify a different column name."
        )


def expects_parameter_value_in_list(key, value, items):
    if value not in items:
        raise OperatorSparkOperatorCustomerError(f"Illegal parameter value. {key} expected to be in {items}, but given {value}")


def expects_parameter_value_in_range(key, value, start, end, nullable=False):
    if nullable and value is None:
        return
    if value is None or (value < start or value > end):
        raise OperatorSparkOperatorCustomerError(
            f"Illegal parameter value. {key} expected to be within range {start} - {end}, but given {value}"
        )


def encode_pyspark_model(model):
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = os.path.join(dirpath, "model")
        # Save the model
        model.save(dirpath)

        # Create the temporary zip-file.
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            # Zip the directory.
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    rel_dir = os.path.relpath(root, dirpath)
                    zf.write(os.path.join(root, file), os.path.join(rel_dir, file))

        zipped = mem_zip.getvalue()
        encoded = base64.b85encode(zipped)
        return str(encoded, "utf-8")


def decode_pyspark_model(model_factory, encoded):
    with tempfile.TemporaryDirectory() as dirpath:
        zip_bytes = base64.b85decode(encoded)
        mem_zip = BytesIO(zip_bytes)
        mem_zip.seek(0)

        with zipfile.ZipFile(mem_zip, "r") as zf:
            zf.extractall(dirpath)

        model = model_factory.load(dirpath)
        return model


def hash_parameters(value):
    try:
        encoded = json.dumps(value, sort_keys=True).encode(encoding="UTF-8", errors="strict")
        return hashlib.sha1(encoded).hexdigest()
    except:  # noqa: E722
        raise RuntimeError("Object not supported for serialization")


def load_trained_parameters(trained_parameters, operator_parameters):
    trained_parameters = trained_parameters if trained_parameters else {}
    parameters_hash = hash_parameters(operator_parameters)
    stored_hash = trained_parameters.get("_hash")
    if stored_hash != parameters_hash:
        trained_parameters = {"_hash": parameters_hash}
    return trained_parameters


def try_decode_pyspark_model(trained_parameters, model_factory, name):
    try:
        model = decode_pyspark_model(model_factory, trained_parameters[name])
        return model, True
    except Exception as e:
        logging.error(f"Could not decode PySpark model {name} from trained_parameters: {e}")
        del trained_parameters[name]
        return None, False


def load_pyspark_model_from_trained_parameters(trained_parameters, model_factory, name):
    if trained_parameters is None or name not in trained_parameters:
        return None, False

    if is_inference_running_mode():
        if isinstance(trained_parameters[name], str):
            model, model_loaded = try_decode_pyspark_model(trained_parameters, model_factory, name)
            if not model_loaded:
                return model, model_loaded
            trained_parameters[name] = model
        return trained_parameters[name], True

    return try_decode_pyspark_model(trained_parameters, model_factory, name)


def fit_and_save_model(trained_parameters, name, algorithm, df):
    model = algorithm.fit(df)
    trained_parameters[name] = encode_pyspark_model(model)
    return model


def transform_using_trained_model(model, df, loaded):
    try:
        return model.transform(df)
    except Exception as e:
        if loaded:
            raise OperatorSparkOperatorCustomerError(
                f"Encountered error while using stored model. Please delete the operator and try again. {e}"
            )
        else:
            raise e


ESCAPE_CHAR_PATTERN = re.compile("[{}]+".format(re.escape(".`")))


def escape_column_name(col):
    """Escape column name so it works properly for Spark SQL"""

    # Do nothing for Column object, which should be already valid/quoted
    if isinstance(col, Column):
        return col

    column_name = col

    if ESCAPE_CHAR_PATTERN.search(column_name):
        column_name = f"`{column_name}`"

    return column_name


def escape_column_names(columns):
    return [escape_column_name(col) for col in columns]


def sanitize_df(df):
    """Sanitize dataframe with Spark safe column names and return column name mappings

    Args:
        df: input dataframe

    Returns:
        a tuple of
            sanitized_df: sanitized dataframe with all Spark safe columns
            sanitized_col_mapping: mapping from original col name to sanitized column name
            reversed_col_mapping: reverse mapping from sanitized column name to original col name
    """

    sanitized_col_mapping = {}
    sanitized_df = df

    for orig_col in df.columns:
        if ESCAPE_CHAR_PATTERN.search(orig_col):
            # create a temp column and store the column name mapping
            temp_col = f"{orig_col.replace('.', '_')}_{temp_col_name(sanitized_df)}"
            sanitized_col_mapping[orig_col] = temp_col

            sanitized_df = sanitized_df.withColumn(temp_col, sanitized_df[f"`{orig_col}`"])
            sanitized_df = sanitized_df.drop(orig_col)

    # create a reversed mapping from sanitized col names to original col names
    reversed_col_mapping = {sanitized_name: orig_name for orig_name, sanitized_name in sanitized_col_mapping.items()}

    return sanitized_df, sanitized_col_mapping, reversed_col_mapping


def add_filename_column(df):
    """Add a column containing the input file name of each record."""
    filename_col_name_prefix = "_data_source_filename"
    filename_col_name = filename_col_name_prefix
    counter = 1
    while filename_col_name in df.columns:
        filename_col_name = f"{filename_col_name_prefix}_{counter}"
        counter += 1
    return df.withColumn(filename_col_name, sf.input_file_name())





IMAGE_PREVIEW_LIMIT = 50


@dataclass
class SupportedContentType(Enum):
    CSV = "CSV"
    PARQUET = "PARQUET"
    TSV = "CSV"
    ORC = "ORC"
    JSON = "JSON"
    JSONL = "JSONL"
    IMAGE = "IMAGE"


@dataclass
class SupportedImageType(Enum):
    """We support all fully supported formats listed here: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html"""

    BLP = "blp"
    BMP = "bmp"
    DDS = "dds"
    DIB = "dib"
    EPS = "eps"
    GIF = "gif"
    ICNS = "icns"
    ICO = "ico"
    IM = "im"
    JPEG = ["jpeg", "jpg", "jpe", "jif"]
    MSP = "msp"
    PCX = "pcx"
    PNG = "png"
    SGI = "sgi"
    SPIDER = "spi"
    TGA = "tga"
    TIFF = ["tiff", "tif"]
    WEBP = "webp"

    @staticmethod
    def list():
        formats = []
        for _format in map(lambda x: x.value, SupportedImageType):
            if isinstance(_format, str):
                formats.append(_format)
            else:
                formats.extend(_format)
        return formats


class S3ObjectType(Enum):
    FILE = "file"
    FOLDER = "folder"


@dataclass
class S3Metadata:
    name: str
    uri: str
    type: str


@dataclass
class S3ObjectMetadata(S3Metadata):
    """A dataclass for modeling a single S3 object metadata.
    """

    content_type: str = None
    size: int = None
    last_modified: str = None


def s3_get_list_objects_response(s3_client, bucket_name, prefix, delimiter, continuation_token=None, max_keys=500):
    request = {
        "Bucket": bucket_name,
        "Delimiter": delimiter,
        "EncodingType": "url",
        "MaxKeys": max_keys,
        "Prefix": prefix,
    }

    if continuation_token:
        request["ContinuationToken"] = continuation_token

    response = s3_client.list_objects_v2(**request)

    return response


def s3_parse_objects(bucket, prefix, response, delimiter):
    objects = []
    if "Contents" not in response:
        return objects
    contents = response["Contents"]
    for obj in contents:
        obj_key = unquote_plus(obj["Key"])
        if (obj_key == prefix or delimiter == "") and not s3_is_file(response, obj_key):
            continue
        obj_name = s3_get_basename(obj_key)
        obj_size = obj["Size"]
        content_type = s3_infer_content_type_v2(uri=obj_key)
        objects.append(
            S3ObjectMetadata(
                name=obj_name,
                uri=s3_format_uri(bucket, obj_key),
                type=S3ObjectType.FILE.value,
                size=obj_size,
                last_modified=str(obj["LastModified"]),
                content_type=content_type,
            )
        )
    return objects


def s3_is_file(response, obj_key):
    try:
        exists = response["CommonPrefixes"]
        return False
    except KeyError:
        if obj_key[-1] == "/":
            return False
        return True


def s3_infer_content_type_v2(uri):
    inferred_content_type: str = PurePath(uri).suffix[1:]
    if inferred_content_type.upper() in SupportedContentType.__members__:
        inferred_content_type = SupportedContentType[inferred_content_type.upper()].value.lower()
    elif inferred_content_type.lower() in SupportedImageType.list():
        inferred_content_type = "IMAGE"
    logging.debug("Inferred content type from file extension is %s", inferred_content_type)
    return inferred_content_type


def s3_format_uri(bucket_name, prefix=""):
    uri = "s3://" + bucket_name + "/" + prefix
    logging.debug("Formatted uri is %s", uri)
    return uri


def s3_get_basename(key):
    basename = PurePath(key).name
    return basename


def s3_parse_bucket_name_and_prefix(uri):
    if uri.startswith("s3a://"):
        uri = uri.replace("s3a://", "s3://")
    parse_result = urlparse(uri)
    bucket_name = parse_result.netloc
    # Replace only the first delimiter and not all as there could be path s3://bucket///folder
    prefix = parse_result.path.replace("/", "", 1)
    return bucket_name, prefix




def type_inference(df):  # noqa: C901 # pylint: disable=R0912
    """Core type inference logic

    Args:
        df: spark dataframe

    Returns: dict a schema that maps from column name to mohave datatype

    """
    columns_to_infer = [escape_column_name(col) for (col, col_type) in df.dtypes if col_type == "string"]

    pandas_df = df[columns_to_infer].toPandas()
    report = {}
    for column_name, series in pandas_df.iteritems():
        column = series.values
        report[column_name] = {
            "sum_string": len(column),
            "sum_numeric": sum_is_numeric(column),
            "sum_integer": sum_is_integer(column),
            "sum_boolean": sum_is_boolean(column),
            "sum_date": sum_is_date(column),
            "sum_datetime": sum_is_datetime(column),
            "sum_null_like": sum_is_null_like(column),
            "sum_null": sum_is_null(column),
        }

    # Analyze
    numeric_threshold = 0.8
    integer_threshold = 0.8
    date_threshold = 0.8
    datetime_threshold = 0.8
    bool_threshold = 0.8

    column_types = {}

    for col, insights in report.items():
        # Convert all columns to floats to make thresholds easy to calculate.
        proposed = MohaveDataType.STRING.value

        sum_is_not_null = insights["sum_string"] - (insights["sum_null"] + insights["sum_null_like"])

        if sum_is_not_null == 0:
            # if entire column is null, keep as string type
            proposed = MohaveDataType.STRING.value
        elif (insights["sum_numeric"] / insights["sum_string"]) > numeric_threshold:
            proposed = MohaveDataType.FLOAT.value
            if (insights["sum_integer"] / insights["sum_numeric"]) > integer_threshold:
                proposed = MohaveDataType.LONG.value
        elif (insights["sum_boolean"] / insights["sum_string"]) > bool_threshold:
            proposed = MohaveDataType.BOOL.value
        elif (insights["sum_date"] / sum_is_not_null) > date_threshold:
            # datetime - date is # of rows with time info
            # if even one value w/ time info in a column with mostly dates, choose datetime
            if (insights["sum_datetime"] - insights["sum_date"]) > 0:
                proposed = MohaveDataType.DATETIME.value
            else:
                proposed = MohaveDataType.DATE.value
        elif (insights["sum_datetime"] / sum_is_not_null) > datetime_threshold:
            proposed = MohaveDataType.DATETIME.value
        column_types[col] = proposed

    for f in df.schema.fields:
        if f.name not in columns_to_infer:
            if isinstance(f.dataType, IntegralType):
                column_types[f.name] = MohaveDataType.LONG.value
            elif isinstance(f.dataType, FractionalType):
                column_types[f.name] = MohaveDataType.FLOAT.value
            elif isinstance(f.dataType, StringType):
                column_types[f.name] = MohaveDataType.STRING.value
            elif isinstance(f.dataType, BooleanType):
                column_types[f.name] = MohaveDataType.BOOL.value
            elif isinstance(f.dataType, TimestampType):
                column_types[f.name] = MohaveDataType.DATETIME.value
            elif isinstance(f.dataType, ArrayType):
                column_types[f.name] = MohaveDataType.ARRAY.value
            elif isinstance(f.dataType, StructType):
                column_types[f.name] = MohaveDataType.STRUCT.value
            else:
                # unsupported types in mohave
                column_types[f.name] = MohaveDataType.OBJECT.value

    return column_types


def _is_numeric_single(x):
    try:
        if isinstance(x, str):
            if "_" in x:
                return False
    except TypeError:
        return False

    try:
        x_float = float(x)
        return np.isfinite(x_float)
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False


def sum_is_numeric(x):
    """count number of numeric element

    Args:
        x: numpy array

    Returns: int

    """
    castables = np.vectorize(_is_numeric_single, otypes=[bool])(x)
    return np.count_nonzero(castables)


def _is_integer_single(x):
    try:
        if not _is_numeric_single(x):
            return False
        return float(x) == int(x)
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False


def sum_is_integer(x):
    castables = np.vectorize(_is_integer_single, otypes=[bool])(x)
    return np.count_nonzero(castables)


def _is_boolean_single(x):
    boolean_list = ["true", "false"]
    try:
        is_boolean = x.lower() in boolean_list
        return is_boolean
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False
    except AttributeError:
        return False


def sum_is_boolean(x):
    castables = np.vectorize(_is_boolean_single, otypes=[bool])(x)
    return np.count_nonzero(castables)


def sum_is_null_like(x):  # noqa: C901
    def _is_empty_single(x):
        try:
            return bool(len(x) == 0)
        except TypeError:
            return False

    def _is_null_like_single(x):
        try:
            return bool(null_like_regex.match(x))
        except TypeError:
            return False

    def _is_whitespace_like_single(x):
        try:
            return bool(whitespace_regex.match(x))
        except TypeError:
            return False

    null_like_regex = re.compile(r"(?i)(null|none|nil|na|nan)")  # (?i) = case insensitive
    whitespace_regex = re.compile(r"^\s+$")  # only whitespace

    empty_checker = np.vectorize(_is_empty_single, otypes=[bool])(x)
    num_is_null_like = np.count_nonzero(empty_checker)

    null_like_checker = np.vectorize(_is_null_like_single, otypes=[bool])(x)
    num_is_null_like += np.count_nonzero(null_like_checker)

    whitespace_checker = np.vectorize(_is_whitespace_like_single, otypes=[bool])(x)
    num_is_null_like += np.count_nonzero(whitespace_checker)
    return num_is_null_like


def sum_is_null(x):
    return np.count_nonzero(pd.isnull(x))


def _is_date_single(x):
    try:
        return bool(date.fromisoformat(x))  # YYYY-MM-DD
    except ValueError:
        return False
    except TypeError:
        return False


def sum_is_date(x):
    return np.count_nonzero(np.vectorize(_is_date_single, otypes=[bool])(x))


def sum_is_datetime(x):
    # detects all possible convertible datetimes, including multiple different formats in the same column
    return pd.to_datetime(x, cache=True, errors="coerce").notnull().sum()


def cast_df(df, schema):
    """Cast dataframe from given schema

    Args:
        df: spark dataframe
        schema: schema to cast to. It map from df's col_name to mohave datatype

    Returns: casted dataframe

    """
    # col name to spark data type mapping
    col_to_spark_data_type_map = {}

    # get spark dataframe's actual datatype
    fields = df.schema.fields
    for f in fields:
        col_to_spark_data_type_map[f.name] = f.dataType
    cast_expr = []

    to_ts = pandas_udf(f=to_timestamp_single, returnType="string")

    # iterate given schema and cast spark dataframe datatype
    for col_name in schema:
        mohave_data_type_from_schema = MohaveDataType(schema.get(col_name, MohaveDataType.OBJECT.value))
        if mohave_data_type_from_schema == MohaveDataType.DATETIME:
            df = df.withColumn(col_name, to_timestamp(to_ts(df[col_name])))
            expr = f"`{col_name}`"  # keep the column in the SQL query that is run below
        elif mohave_data_type_from_schema != MohaveDataType.OBJECT:
            spark_data_type_from_schema = MOHAVE_TO_SPARK_TYPE_MAPPING.get(mohave_data_type_from_schema)
            if not spark_data_type_from_schema:
                raise KeyError(f"Key {mohave_data_type_from_schema} not present in MOHAVE_TO_SPARK_TYPE_MAPPING")
            # Only cast column when the data type in schema doesn't match the actual data type
            # and data type is not Array or Struct
            if spark_data_type_from_schema not in [ArrayType, StructType] and not isinstance(
                col_to_spark_data_type_map[col_name], spark_data_type_from_schema
            ):
                # use spark-sql expression instead of spark.withColumn to improve performance
                expr = f"CAST (`{col_name}` as {SPARK_TYPE_MAPPING_TO_SQL_TYPE[spark_data_type_from_schema]})"
            else:
                # include column that has same dataType as it is
                expr = f"`{col_name}`"
        else:
            # include column that has same mohave object dataType as it is
            expr = f"`{col_name}`"
        cast_expr.append(expr)
    if len(cast_expr) != 0:
        df = df.selectExpr(*cast_expr)
    return df, schema


def validate_schema(df, schema):
    """Validate if every column is covered in the schema

    Args:
        schema ():
    """
    columns_in_df = df.columns
    columns_in_schema = schema.keys()

    if len(columns_in_df) != len(columns_in_schema):
        raise ValueError(
            f"Invalid schema column size. "
            f"Number of columns in schema should be equal as number of columns in dataframe. "
            f"schema columns size: {len(columns_in_schema)}, dataframe column size: {len(columns_in_df)}"
        )

    for col in columns_in_schema:
        if col not in columns_in_df:
            raise ValueError(
                f"Invalid column name in schema. "
                f"Column in schema does not exist in dataframe. "
                f"Non-existed columns: {col}"
            )


def s3_source(spark, mode, dataset_definition, flow_parameters=None):
    """Represents a source that handles sampling, etc."""
    import boto3

    # s3 client for imports that require s3
    s3_client = boto3.client("s3")


    path = dataset_definition["s3ExecutionContext"]["s3Uri"].replace("s3://", "s3a://")

    content_type = dataset_definition["s3ExecutionContext"]["s3ContentType"].upper()
    recursive = "true" if dataset_definition["s3ExecutionContext"].get("s3DirIncludesNested") else "false"
    adds_filename_column = dataset_definition["s3ExecutionContext"].get("s3AddsFilenameColumn", False)
    role_arn = dataset_definition["s3ExecutionContext"].get("s3RoleArn", None)

    try:
            if content_type == "CSV":
                has_header = dataset_definition["s3ExecutionContext"]["s3HasHeader"]
                field_delimiter = dataset_definition["s3ExecutionContext"].get("s3FieldDelimiter", ",")
                if not field_delimiter:
                    field_delimiter = ","
                df = spark.read.option("recursiveFileLookup", recursive).csv(
                    path=path, header=has_header, escape='"', quote='"', sep=field_delimiter, mode="PERMISSIVE"
                )
            elif content_type == SupportedContentType.PARQUET.value:
                df = spark.read.option("recursiveFileLookup", recursive).parquet(path)
            elif content_type == SupportedContentType.JSON.value:
                df = spark.read.option("multiline", "true").option("recursiveFileLookup", recursive).json(path)
            elif content_type == SupportedContentType.JSONL.value:
                df = spark.read.option("multiline", "false").option("recursiveFileLookup", recursive).json(path)
            elif content_type == SupportedContentType.ORC.value:
                df = spark.read.option("recursiveFileLookup", recursive).orc(path)
            elif content_type == SupportedContentType.IMAGE.value:
                # TODO: make this lazy using rdds and maps: https://issues.amazon.com/issues/SDW-3477
                bucket_name, prefix = s3_parse_bucket_name_and_prefix(path)
                delimiter = "" if recursive == "true" else "/"
                response = s3_get_list_objects_response(s3_client, bucket_name, prefix, delimiter)
                objects = s3_parse_objects(bucket=bucket_name, prefix=prefix, response=response, delimiter=delimiter)
                object_list = [[obj.uri] for obj in objects if obj.content_type == SupportedContentType.IMAGE.value]
                df = spark.createDataFrame(object_list, ["origin"],)
                image_rdd = make_image_rdd(df, mode)
                return default_spark(image_rdd)
            if adds_filename_column:
                df = add_filename_column(df)
            return default_spark(df)
    except Exception as e:
        raise RuntimeError("An error occurred while reading files from S3") from e


def infer_and_cast_type(df, spark, inference_data_sample_size=1000, trained_parameters=None):
    """Infer column types for spark dataframe and cast to inferred data type.

    Args:
        df: spark dataframe
        spark: spark session
        inference_data_sample_size: number of row data used for type inference
        trained_parameters: trained_parameters to determine if we need infer data types

    Returns: a dict of pyspark df with column data type casted and trained parameters

    """

    # if trained_parameters is none or doesn't contain schema key, then type inference is needed
    if trained_parameters is None or not trained_parameters.get("schema", None):
        # limit first 1000 rows to do type inference
        limit_df = df.limit(inference_data_sample_size)
        schema = type_inference(limit_df)
    else:
        schema = trained_parameters["schema"]
        try:
            validate_schema(df, schema)
        except ValueError as e:
            raise OperatorCustomerError(e)
    try:
        df, schema = cast_df(df, schema)
    except (AnalysisException, ValueError) as e:
        raise OperatorCustomerError(e)
    trained_parameters = {"schema": schema}
    return default_spark_with_trained_parameters(df, trained_parameters)


def handle_missing(df, spark, **kwargs):

    # Handle the old interface for Drop missing by converting to new interface
    if kwargs["operator"] == "Drop missing":
        drop_missing_params = kwargs.get("drop_missing_parameters")
        drop_rows_params = drop_missing_params.get("drop_rows_parameters") if drop_missing_params else None
        if drop_rows_params and drop_missing_params.get("dimension") == "Drop Rows":
            kwargs["drop_missing_parameters"] = drop_rows_params

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Impute": (handle_missing_impute, "impute_parameters"),
            "Fill missing": (handle_missing_fill_missing, "fill_missing_parameters"),
            "Add indicator for missing": (
                handle_missing_add_indicator_for_missing,
                "add_indicator_for_missing_parameters",
            ),
            "Drop missing": (handle_missing_drop_rows, "drop_missing_parameters"),
        },
    )


def cast_single_data_type(  # noqa: C901
    df,
    spark,
    column,
    data_type,
    non_castable_data_handling_method="replace_null",
    replace_value=None,
    date_formatting="dd-MM-yyyy",
    datetime_formatting=None,
):
    """Cast pyspark dataframe column type

    Args:
        column: column name e.g.: "col_1"
        data_type: data type to cast to
        non_castable_data_handling_method:
            supported method:
                ("replace_null","replace_null_with_new_col", "replace_value","replace_value_with_new_col","drop")
            If not specified, it will use the default method replace_null.
            see casting.NonCastableDataHandlingMethod
        replace_value: value to replace non-castable data
        date_formatting: date format to cast to
        datetime_formatting: datetime format to cast to

    Returns: df: pyspark df with column data type casted
    """
    from pyspark.sql.utils import AnalysisException

    supported_type = MohaveDataType.get_values()
    df_cols = df.columns
    # Validate input params
    if column not in df_cols:
        raise OperatorCustomerError(
            f"Invalid column name. {column} is not in current columns {df_cols}. Please use a valid column name."
        )
    if data_type not in supported_type:
        raise OperatorCustomerError(
            f"Invalid data_type. {data_type} is not in {supported_type}. Please use a supported data type."
        )

    support_invalid_data_handling_method = NonCastableDataHandlingMethod.get_values()
    if non_castable_data_handling_method not in support_invalid_data_handling_method:
        raise OperatorCustomerError(
            f"Invalid data handling method. "
            f"{non_castable_data_handling_method} is not in {support_invalid_data_handling_method}. "
            f"Please use a supported method."
        )

    mohave_data_type = MohaveDataType(data_type)

    spark_data_type = [f.dataType for f in df.schema.fields if f.name == column]

    if isinstance(spark_data_type[0], MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type]):
        return default_spark(df)

    try:
        df = cast_single_column_type(
            df,
            column=column,
            mohave_data_type=MohaveDataType(data_type),
            invalid_data_handling_method=NonCastableDataHandlingMethod(non_castable_data_handling_method),
            replace_value=replace_value,
            date_formatting=date_formatting,
            datetime_formatting=datetime_formatting,
        )
    except (AnalysisException, ValueError) as e:
        raise OperatorCustomerError(e)

    return default_spark(df)


op_1_output = s3_source(spark=spark, mode=mode, **{'dataset_definition': {'__typename': 'S3CreateDatasetDefinitionOutput', 'datasetSourceType': 'S3', 'name': 'titanic.csv', 'description': None, 's3ExecutionContext': {'__typename': 'S3ExecutionContext', 's3Uri': 's3://udacitydatatitanic/titanic.csv', 's3ContentType': 'csv', 's3HasHeader': True, 's3FieldDelimiter': ',', 's3DirIncludesNested': False, 's3AddsFilenameColumn': False}}})
op_2_output = infer_and_cast_type(op_1_output['default'], spark=spark, **{})
op_3_output = handle_missing(op_2_output['default'], spark=spark, **{'operator': 'Impute', 'impute_parameters': {'column_type': 'Numeric', 'numeric_parameters': {'strategy': 'Mean', 'input_column': ['Age'], 'output_column': 'AgeImputed'}}})
op_4_output = cast_single_data_type(op_3_output['default'], spark=spark, **{'column': 'Survived', 'original_data_type': 'Long', 'data_type': 'bool'})

#  Glossary: variable name to node_id
#
#  op_1_output: 30730182-a9a8-465d-8798-aacdd1fd5b20
#  op_2_output: a633b2cf-6076-4db3-97d5-b3505849c9af
#  op_3_output: 5b1c469f-1da7-4d40-b17c-9b198b3ff7be
#  op_4_output: 6a2c74f6-79cf-4ea0-ac14-d4249dd9ef33