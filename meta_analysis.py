from argparse import ArgumentParser
from functools import reduce

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, desc
from pyspark.sql.types import StructType


def main():
    parser = ArgumentParser()
    parser.add_argument('--spark-connect-url', '-S', type=str, default='sc://localhost:15002')
    parser.add_argument('--output-dir', '-o', type=str, default='./output')
    parser.add_argument('input_files', nargs='+')
    args = parser.parse_args()

    spark = SparkSession.builder.remote(args.spark_connect_url).getOrCreate()

    post_schema = StructType() \
        .add('topic', 'string', False) \
        .add('user', 'string', False) \
        .add('post', 'string', False) \
        .add('sentiment', 'string', False)
    posts = reduce(lambda acc, curr: acc.union(curr),
                   (spark.read.csv(file, header=False, schema=post_schema) for file in args.input_files))
    posts = posts.distinct().filter(col('sentiment').rlike(r'^..$'))
    total_posts = posts.count()
    global_dist = posts.groupby('sentiment').agg(count('*').alias('count')) \
        .withColumn('ratio', col('count') / total_posts)
    global_dist.select('sentiment', 'ratio').show()

    topic_count = posts.groupby('topic').agg(count('*').alias('count'))
    topic_dist = posts.groupby('topic', 'sentiment').agg(count('post').alias('count'))
    topic_dist = topic_dist.join(topic_count.select(col('topic'), col('count').alias('topic_size')),
                                 topic_dist.topic == topic_count.topic)
    topic_dist = topic_dist.select(topic_count.topic, col('sentiment'), col('topic_size'),
                                   (col('count') / col('topic_size')).alias('ratio')) \
        .sort(desc('topic_size'))
    topic_dist.show()

    global_dist.write.csv(f'{args.output_dir}/weebiee_global_dist')
    topic_dist.write.csv(f'{args.output_dir}/weebiee_topic_dist')


if __name__ == '__main__':
    main()
