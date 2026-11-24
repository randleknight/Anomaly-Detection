## This needs to be defined

import featureform as ff

postgres = ff.register_postgres(
    name="postgres-quickstart",
    host="host",  # The docker dns name for postgres
    port="5432",
    user="postgres",
    password="password",
    database="database",
)

redis = ff.register_redis(
    name = "redi",
    host="host.docker.internal", # The docker dns name for redis
    port=6379,
)

transactions = postgres.register_table(
    name="result",
    table="result",  # This is the table's name in Postgres
)

@postgres.sql_transformation(inputs=[transactions])
def average_user_transaction(tr):
    return "select sample_number , count(1) as number_tests  from {{tr}} r group by sample_number " 


@ff.entity
class User:
    avg_transactions = ff.Feature(
        average_user_transaction[["sample_number", "number_tests"]], # We can optional include the `timestamp_column` "timestamp" here
        type=ff.Float32,
        inference_store=redis,
    )
    fraudulent = ff.Label(
        transactions[["sample_number","number_tests"]], variant="quickstart", type=ff.Bool
    )

ff.register_training_set(
    "fraud_training",
    label=("fraudulent", "quickstart"),
    features=["avg_transactions"],
)

@ff.entity
class User:
    avg_transactions = ff.Variants(
        {
            "quickstart": ff.Feature(
                average_user_transaction[["sample_number", "number_tests"]],
                type=ff.Float32,
                inference_store=redis,
            ),
            "quickstart_v2": ff.Feature(
                average_user_transaction[["sample_number", "number_tests"]],
                type=ff.Float32,
                inference_store=redis,
            ),
        }
    )
    fraudulent = ff.Label(
        transactions[["sample_number","number_tests"]], variant="quickstart", type=ff.Bool
    )
