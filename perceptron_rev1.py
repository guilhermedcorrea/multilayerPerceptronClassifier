from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from functools import wraps

input_cols = ["vendaid", "pessoaId", "numerodepequenasemicro", "ocupacao", "salarios", 
              "valoradicionado", "TAXACRESCIMENTO10", "TAXACRESCIMENTOATE40", 
              "TAXACRESCIMENTOATE50"]


def convert_columns_to_double(columns):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            for col_name in columns:
                self = self.withColumn(col_name, col(col_name).cast(DoubleType()))
            return func(self, *args[1:], **kwargs)
        return wrapper
    return decorator


def apply_one_hot_encoder(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        non_numeric_cols = [f.name for f in self.schema.fields
                            if not isinstance(f.dataType, (DoubleType,))]
        for col_name in non_numeric_cols:
            if col_name in input_cols:
                encoder = OneHotEncoder(inputCols=[col_name], outputCols=[f"{col_name}_encoded"])
                self = encoder.fit(self).transform(self).drop(col_name)
        return func(self, *args[1:], **kwargs)
    return wrapper

def add_label_column(label_col):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            self = self.withColumn("label", col(label_col))
            return func(self, *args[1:], **kwargs)
        return wrapper
    return decorator

@convert_columns_to_double(input_cols)
@apply_one_hot_encoder
@add_label_column("valores")
def preprocess_and_train(data, input_cols):
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    data = assembler.transform(data)

    layers = [len(input_cols), 5, 4, 3]

    train, test = data.randomSplit([0.6, 0.4], 1234)

    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234, labelCol="label")

    model = trainer.fit(train)

    result = model.transform(test)

    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator.evaluate(result.select("prediction", "label"))
    print("Teste acuracia =", accuracy)

spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()

data = spark.read.options(header='True', inferSchema='True', delimiter=';') \
    .csv("/home/guilherme/spark_analytics/teste0608.csv")

preprocess_and_train(data, input_cols)

spark.stop()
