package ikkyu.cfo.builddata;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import nlp.tokenizer.EnglishTokenizer;

public class BuildSubjectData {
  // Input files
  private static final String FREEBASE_SUBJECT_NAME_PATH = "../data/transE/FB5M-entity-id.txt";
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

  private static void buildSubjectData(String question_path, String subject_name_path,
      String oPath) throws IOException {
    // Load subject-id map
    File fin = new File(subject_name_path);
    BufferedReader reader = new BufferedReader(new FileReader(fin));
    Map<String, String> subject_id_map = new HashMap<String, String>();
    String line;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.trim().split("\t");
      String subject_name = tokens[0].trim().replace("www.freebase.com", "");
      String subject_id = tokens[1].trim();
      if (!subject_id_map.containsKey(subject_name)) {
        subject_id_map.put(subject_name, subject_id);
      }
    }
    reader.close();

    File fout = new File(oPath);
    BufferedWriter writer = new BufferedWriter(new FileWriter(fout));
    fin = new File(question_path);
    reader = new BufferedReader(new FileReader(fin));
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.trim().split("\t");

      String subject_name = tokens[0].trim().replace("www.freebase.com", "");
      if (subject_id_map.get(subject_name) == null) {
        continue;
      }
      String subject_id = subject_id_map.get(subject_name);
      
      String question = tokens[3].trim();
      List<String> question_tokens = EnglishTokenizer.tokenizer(question.toLowerCase());
      writer.write(
          question_tokens.stream().collect(Collectors.joining(" ")) + "\t" + subject_id + "\n");
    }
    reader.close();
    writer.close();
  }

  public static void main(String[] args) throws IOException {
    buildSubjectData(TRAIN_FILE_PATH, FREEBASE_SUBJECT_NAME_PATH, SUBJECT_TRAIN_FILE_PATH);
    buildSubjectData(VALID_FILE_PATH, FREEBASE_SUBJECT_NAME_PATH, SUBJECT_VALID_FILE_PATH);
    buildSubjectData(TEST_FILE_PATH, FREEBASE_SUBJECT_NAME_PATH, SUBJECT_TEST_FILE_PATH);
  }

}
