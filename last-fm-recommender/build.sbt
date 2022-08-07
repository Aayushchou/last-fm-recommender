val scala3Version = "2.12.11"

lazy val root = project
  .in(file("."))
  .settings(
    name := "last-fm-recommender",
    version := "0.1.0-SNAPSHOT",

    scalaVersion := scala3Version,

  libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-mllib" % "3.0.0",
  "org.apache.spark" %% "spark-sql" % "3.0.0",
  "com.github.scopt" %% "scopt" % "4.1.0",
  "org.scalameta" %% "munit" % "0.7.29" % Test
// comment above lines and uncomment the following to run in sbt console
// "org.apache.spark" %% "spark-streaming" % "1.6.1",
// "org.apache.spark" %% "spark-mllib" % "1.6.1"
  )
  )