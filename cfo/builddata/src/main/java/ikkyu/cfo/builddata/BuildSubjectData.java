package ikkyu.cfo.builddata;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.List;
import java.util.stream.Collectors;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

import KnowledgeGraph.KGSearch.EntitySearch;
import KnowledgeGraph.KGSearch.EntitySearch.Pair;
import nlp.tokenizer.EnglishTokenizer;

public class BuildSubjectData {
  // Input files
  private static final String TRAIN_FILE_PATH =
      "../../data/SimpleQuestions_v2/annotated_fb_data_train.txt";
  private static final String VALID_FILE_PATH =
      "../../data/SimpleQuestions_v2/annotated_fb_data_valid.txt";
  private static final String TEST_FILE_PATH =
      "../../data/SimpleQuestions_v2/annotated_fb_data_test.txt";

  // Output files
  private static final String SUBJECT_TRAIN_FILE_PATH = "../data/subject/subject_train.txt";
  private static final String SUBJECT_VALID_FILE_PATH = "../data/subject/subject_valid.txt";
  private static final String SUBJECT_TEST_FILE_PATH = "../data/subject/subject_test.txt";

  // KGSearch
  private static final EntitySearch entity_search = EntitySearch.getEntitySearch();

  private static void buildSubjectData(String question_path, String oPath) throws Exception {
    Jedis jedis = null;
    JedisPool pool = new JedisPool(new JedisPoolConfig(), "localhost");
    try {
      int cnt = 0;

      File fout = new File(oPath);
      BufferedWriter writer = new BufferedWriter(new FileWriter(fout));
      jedis = pool.getResource();
      File fin = new File(question_path);
      BufferedReader reader = new BufferedReader(new FileReader(fin));
      String line;
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.trim().split("\t");
        String question = tokens[3].trim();
        List<String> question_tokens = EnglishTokenizer.tokenizer(question.toLowerCase());
        question = question_tokens.stream().collect(Collectors.joining(" "));
        String subject_mid = tokens[0].trim().replace("www.freebase.com", "");

        // Positive sample
        jedis.select(0);
        String id = jedis.get(subject_mid);
        if (id != null) {
          jedis.select(2);
          String vector = jedis.get(id);
          writer.write(question + "\t1\t" + vector + "\n");
        }

        // Negative samples
        jedis.select(1);
        String name = jedis.get(subject_mid);
        if (name == null) {
          continue;
        }
        if (name.length() > 0 && name.charAt(0) == '\"' && name.charAt(name.length() - 1) == '\"') {
          name = name.substring(1, name.length() - 1);
        }
        List<Pair> pair_list = entity_search.search(name);
        for (Pair p : pair_list) {
          String neg_sample_mid = p.mid;
          if (neg_sample_mid.equals(subject_mid)) {
            ++cnt;
            continue;
          }
          jedis.select(0);
          String neg_id = jedis.get(neg_sample_mid);
          if (neg_id != null) {
            jedis.select(2);
            String neg_vector = jedis.get(neg_id);
            writer.write(question + "\t0\t" + neg_vector + "\n");
          }
        }
      }
      System.out.println("There are " + cnt + " results from Google API containg our correct mid");
      reader.close();
      writer.close();
    } finally {
      if (jedis != null) {
        jedis.close();
      }
    }
  }

  public static void main(String[] args) throws Exception {
    buildSubjectData(TRAIN_FILE_PATH, SUBJECT_TRAIN_FILE_PATH);
    buildSubjectData(VALID_FILE_PATH, SUBJECT_VALID_FILE_PATH);
    buildSubjectData(TEST_FILE_PATH, SUBJECT_TEST_FILE_PATH);

  }

}
