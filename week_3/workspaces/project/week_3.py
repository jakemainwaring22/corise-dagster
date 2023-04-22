from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    RetryPolicy,
    RunRequest,
    ScheduleDefinition,
    SensorEvaluationContext,
    SkipReason,
    String,
    graph,
    op,
    schedule,
    sensor,
    static_partitioned_config,
)
from workspaces.config import REDIS, S3
from workspaces.project.sensors import get_s3_keys
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock


@op(
    config_schema={"s3_key": String},
    required_resource_keys={"s3"})
def get_s3_data(context: OpExecutionContext):
    # Pulling S3 file path from config and returning list of stocks 
    s3_key = context.op_config["s3_key"]
    s3_resource_info = context.resources.s3
    raw_data = s3_resource_info.get_data(key_name=s3_key)
    stock_list = []
    for stock in raw_data:
        stock_list.append(Stock.from_list(stock))
    return stock_list


@op(
    ins={"stocks": In(dagster_type=List, description="List of stocks to be processed")},
    out={"agg": Out(dagster_type=Aggregation, description="Stock with the max high and its date")}
    )
def process_data(stocks):
    # Pulling date and high from stock with the max high price   
    max_stock = max(stocks, key=lambda x: x.high)
    agg = Aggregation(date=max_stock.date, high=max_stock.high)    
    return agg


@op(
    ins={"agg": In(dagster_type=Aggregation, description="Aggregated data with maximum stock high and date")},
    required_resource_keys={"redis"},
    out=Out(dagster_type=Nothing)
    )
def put_redis_data(context, agg):
    # Getting date and high price info
    date_of_max = str(agg.date)
    high_of_max = str(agg.high)
    # Creating a redis client and writing to redis
    redis_client = context.resources.redis
    redis_client.put_data(name=date_of_max, value=high_of_max)


@op(
    ins={"agg": In(dagster_type=Aggregation, description="Aggregated data with maximum stock high and date")},
    required_resource_keys={"s3"},
    out=Out(dagster_type=Nothing)
    )
def put_s3_data(context, agg):
    # Getting date and high price info
    date_of_max = str(agg.date)
    high_of_max = str(agg.high)
    # Creating an S3 client and writing to S3
    s3_client = context.resources.s3
    s3_client.put_data(key_name=date_of_max, data=high_of_max)


@graph
def machine_learning_graph():
    stock_data = get_s3_data()
    max_stock_info = process_data(stock_data)
    put_redis_data(max_stock_info)
    put_s3_data(max_stock_info)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}


docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}

# Create array of static partition names then pass in to generate config for each static partition
partition_array = [str(elem) for elem in range(1, 11)]
@static_partitioned_config(partition_keys=partition_array)
def docker_config(partition_key: str):
    return {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": f"prefix/stock_{partition_key}.csv"}}},
}


# Job that does not use Docker
machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()}  
)

# Job that uses Docker. Retry policy applies to failure of any op within the job. Ten retries until job fails
machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker_config,
    resource_defs={"s3": s3_resource, "redis": redis_resource},
    op_retry_policy=RetryPolicy(max_retries=10, delay=1)
)

# Creating schedule for machine_learning_job_local that runs every fifteen minutes
machine_learning_schedule_local = ScheduleDefinition(cron_schedule="*/15 * * * *", job=machine_learning_job_local)


# Creating schedule for machine_learning_job_docker that runs at the top of every hour, for each partition
@schedule(cron_schedule="0 * * * *", job=machine_learning_job_docker)
def machine_learning_schedule_docker():
    pass


@sensor(job=machine_learning_job_docker)
def machine_learning_sensor_docker(context):
    # Return new s3 keys since last time it was called
    new_s3_keys = get_s3_keys(bucket='dagster', prefix='prefix', endpoint_url='http://localstack:4566/', max_keys=10)
  
    # Skip if there are no new s3 files (new_s3_keys array is empty)
    # Yield rather than return since return would stop the sensor function from running altogether after being called
    if not new_s3_keys:
        yield SkipReason("No new s3 files found in bucket.")

    # Case where there are new s3 files since last checking
    # Trigger a new run request for each new file in there. Run_key identifies the run
    else:
        for s3_key in new_s3_keys:
            yield RunRequest(
                run_key=s3_key,
                run_config={
                    "resources": {
                        "s3": {"config": S3},
                        "redis": {"config": REDIS},
                    },
                    "ops": {
                        "get_s3_data": {"config": {"s3_key": s3_key}}
                    },
                }
        )
