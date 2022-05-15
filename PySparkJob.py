import io
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff
from pyspark.sql import functions as F


def process(spark, input_file, target_path):
    """
    Функция обработки спарк датафрема.
    Датафрейм разделяется на трейн и тест выборку и сохраняется в файл parquet.
    @param spark: объект спарк
    @param input_file: входящий файл parquet
    @param target_path: путь для сохранения файлов
    """
    # загружаем датафрем
    df = spark.read.parquet(input_file)

    # кодируем фичу событий
    df_event_ohe = feature_to_one_hot(df, "event")

    # группируем датафрем ad_id
    df_grouped = group_by_ad_id(df_event_ohe)

    # кодируем фичу типа рекламы
    df_grouped_new_feat = feature_to_one_hot(df_grouped, "ad_cost_type").drop(
        "ad_cost_type"
    )

    # добавляем фичу количества дней
    df_grouped_new_feat = add_day_count(df_grouped_new_feat)

    # добавляем фичу ctr
    df_grouped_new_feat = add_ctr(df_grouped_new_feat)

    splits = df_grouped_new_feat.randomSplit([0.75, 0.25], 42)

    train, test = splits[0], splits[1]

    train.write.parquet(target_path + '/train')
    test.write.parquet(target_path + "/test")


def feature_to_one_hot(df, col_name):
    """
    Функция для кодирования категориальной переменной методо one hot encoding.
    @param df: спарк датафрейм
    @param col_name: название колонки
    @return: спарк датафрейм
    """
    uniq_values = [row[col_name] for row in df.select(col_name).distinct().collect()]

    for value in uniq_values:
        df = df.withColumn(
            f"is_{value}".lower(), (col(col_name) == value).cast("integer")
        )

    return df


def group_by_ad_id(df, by="ad_id"):
    """
    Функция для группировки данных по ad_id и агрегации признаков.
    @param df: спарк датафрейм
    @param by: название колонкин содержащей ad_id
    @return: спарк датафрейм
    """
    df = df.groupby(by).agg(
        F.first("ad_cost_type").alias("ad_cost_type"),
        F.first("has_video").alias("has_video"),
        F.sum("target_audience_count").alias("target_audience_count"),
        F.min("date"),
        F.max("date"),
        F.sum("is_view").alias("views_cnt"),
        F.sum("is_click").alias("clicks_cnt"),
        F.round(F.sum("ad_cost")).alias("ad_cost"),
    )
    return df


def add_day_count(df):
    """
    Функция для добавления колонки с числом дней, которое показвалась реклама.
    @param df: спарк датафрейм
    @return: спарк датафрейм
    """
    df = df.withColumn(
        "day_count", F.datediff(col("max(date)"), col("min(date)"))
    ).drop("max(date)", "min(date)")
    return df


def add_ctr(df):
    """
    Функция для добавления колонки CTR.
    @param df: спарк датафрейм
    @return: спарк датафрейм
    """
    df = df.withColumn("CTR", F.round(col("clicks_cnt") / col("views_cnt"), 4)).drop(
        "views_cnt", "clicks_cnt"
    )

    return df


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName("PySparkJob").getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
