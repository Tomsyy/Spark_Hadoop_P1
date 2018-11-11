package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit, TrainValidationSplitModel}



object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")

    /** ------------------------------------------------------------------------------------------------------------ **/

    /** 1 - CHARGEMENT DES DONNEES **/

    // on affiche les 5 premières lignes du DataFrame

    val df:DataFrame = spark.read.parquet("./src/prepared_trainingset")
    df.show(5)

    /** ------------------------------------------------------------------------------------------------------------ **/

    /** 2 - DONNEES TEXT **/

    //2-a)

    /** RegexTokenizer prend un texte est le décompose en mots **/

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //2-b)

    /**  StopWordsRemover permet de filtrer une liste de mots **/

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")


    // 2-c)

    /** CountVectorizer compte les occurences des tokens : partie TF **/

    val vectorize = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("TF")

    // 2-d)

    /** IDF est un estimateur qui produit un IDFmodel : partie IDF **/

    val tfidf = new IDF()
      .setInputCol("TF")
      .setOutputCol("tfidf")

    /** ----------------------------------------------------------------------------------------------------------- **/

    /** 3 Convertir les catégories en données numériques **/

    //3-e)

    /** StringIndexer donne un indice pour chaque terme/etiquette selon les occurences **/

    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip") // pour éviter les erreurs de compilation

    //3-f)

    /** StringIndexer donne un indice pour chaque terme/etiquette selon les occurences **/

    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip") // pour éviter les erreurs de compilation

    //3-g)

    /**
      * OneHotEncoder permet un mapping, utile pour les regression logistic
      * /!\ n'existe plus dans Spark 3.0.0 --> OneHotEncoderEstimator
      **/

    val currency_encoder = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_hot")

    val country_encoder = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_hot")

    /** ----------------------------------------------------------------------------------------------------------- **/

    /** 4 Mettre les données sous une forme utilisable par Spark.ML **/

    //4-h)

    /** VectorAssembler assemble des features ensemble **/
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_hot", "currency_hot"))
      .setOutputCol("features")


    //4-i)

    /** mise en place de la regression logistic **/

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setTol(1.0e-6)
      .setMaxIter(300)

    //4-j)

    /** Pipeline configure une pipeline ML en assemblant les "stages" **/

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, vectorize, tfidf, indexerCountry, indexerCurrency,
        currency_encoder, country_encoder, assembler, lr)) // 11 stages car l'encoder est divisé en 2 stages

    /** ----------------------------------------------------------------------------------------------------------- **/

    /** 5 Entraînement et tuning du modèle **/

    //5-k)

    /** randomSplit divise le DataFrame en groupe d'entrainement et de test, seed = évite l'aléa **/

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 2)

    //5-l

    /** ParamGridBuilder crée une grille de valeur avec les hyper-paramètres que l'on souhaite tester **/

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
     // .addGrid(lr.regParam, 10e-8 to 10e-2 by 2.0)
      .addGrid(vectorize.minDF, Array(55.0, 75.0, 95.0))
     // .addGrid(vectorize.minDF, 55.0 to 95.0 by 20.0)
      .build()
    // les deux méthodes fonctionnent

    /** MulticlassClassificationEvaluator on effectue une classification selon le F1-Score **/

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1") // valeur par défaut

    /** TrainValidationSplit permet d'obtenir les hyper-paramètres les plus efficaces **/

    val trainValidation = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    /** on applique le modèle, normalement optimal, au training set **/
    val model_fit = trainValidation.fit(training)

      // Test du modèle

    //5-m)

    /** on applique le modèle aux données de Test **/

    val df_WithPredictions = model_fit.transform(test)

    /** affichage du f-score **/

    print("f-score : "+evaluator.evaluate(df_WithPredictions)+"\n")

    //5-n)
    df_WithPredictions
      .groupBy("final_status", "predictions")
      .count()
      .show()

    // Now we can optionally save the fitted pipeline to disk
    model_fit.write.overwrite().save("/home/thomas/Bureau/MS Telecom Paristech/Cours/INF729_Introduction_framework_HadoopSpark/Model")
  }
}