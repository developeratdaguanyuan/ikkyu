package ikkyu.cfo.builddata;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import nlp.tokenizer.EnglishTokenizer;

public class FocusBuildData {  
  // Input files
  private static final String FREEBASE_ENTITY_NAME_PATH =
	  "../../data/entity_tokenizedname_map_bak.txt";
  private static final String TRAIN_FILE_PATH =
      "../../data/SimpleQuestions_v2/annotated_fb_data_train.txt";
  private static final String VALID_FILE_PATH =
      "../../data/SimpleQuestions_v2/annotated_fb_data_valid.txt";
  private static final String TEST_FILE_PATH =
      "../../data/SimpleQuestions_v2/annotated_fb_data_test.txt";

  // Output files
  private static final String FOCUS_TRAIN_FILE_PATH = "../focus/data/focus_train.txt";
  private static final String FOCUS_VALID_FILE_PATH = "../focus/data/focus_valid.txt";
  private static final String FOCUS_TEST_FILE_PATH = "../focus/data/focus_test.txt";
  
  
  private static void getSentenceFocus(String question_path, String entity_name_path, String oPath)
      throws IOException {
    // Load subject id in question_path to HashMap "buffer"
    File fin = new File(question_path);
    BufferedReader reader = new BufferedReader(new FileReader(fin));
    Map<String, String> buffer = new HashMap<String, String>();
    String line;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.trim().split("\t");
      String sub_id = tokens[0].trim().replace("www.freebase.com", "");
      if (!buffer.containsKey(sub_id)) {
        buffer.put(sub_id, "");
      }
    }
    reader.close();

    // Load subject name from entity_name_path to HashMap "buffer"
    fin = new File(entity_name_path);
    reader = new BufferedReader(new FileReader(fin));
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.trim().split("\t");
      if (2 != tokens.length) {
        continue;
      }
      String entity = tokens[0].trim();
      String name = tokens[1].trim();
      if (buffer.containsKey(entity)) {
        buffer.put(entity, name);
      }
    }
    reader.close();
    
    // Generate data for Focus to buf_for_sort
    List<String> buf_for_sort = new ArrayList<String>();
    fin = new File(question_path);
    reader = new BufferedReader(new FileReader(fin));
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.trim().split("\t");
      String sub_id = tokens[0].trim().replace("www.freebase.com", "");
      String question = tokens[3].trim();
      if (buffer.containsKey(sub_id)) {
        boolean label = false;
        List<String> question_tokens = EnglishTokenizer.tokenizer(question.toLowerCase());
        List<String> sub_tokens = Arrays.asList(buffer.get(sub_id).split(" "));
        //List<String> question_tokens = Arrays.asList(question.split(" "));
        for (int i = 0; i < question_tokens.size(); i++) {
          int q = i, f = 0;
          while (q < question_tokens.size() && f < sub_tokens.size()) {
            if (!question_tokens.get(q).equals(sub_tokens.get(f))) {
              break;
            }
            ++q;
            ++f;
          }
          if (f == sub_tokens.size()) {
            String[] mark = new String[question_tokens.size()];
            Arrays.fill(mark, "0");
            for (int j = i; j < q; j++) {
              mark[j] = "1";
            }

            buf_for_sort.add(question_tokens.stream().collect(Collectors.joining(" "))
                + "\t" + Arrays.asList(mark).stream().collect(Collectors.joining(" ")));
            label = true;
            break;
          }
        }
        if (!label) {
          if (buffer.get(sub_id).equals(""))
            System.out.println(sub_id + "\t" + buffer.get(sub_id) + "\t" + question);
        }
      } else {
        System.out.println("Error!!!");
      }
    }
    reader.close();
    
    // Sort buf_for_sort and write it to file
    Collections.sort(buf_for_sort, new Comparator<String>() {
      public int compare(String s1, String s2) {
        String[] s1_array = s1.split("\t")[0].split(" ");
        String[] s2_array = s2.split("\t")[0].split(" ");
        return s2_array.length - s1_array.length;
      }
    });
    File fout = new File(oPath);
    BufferedWriter writer = new BufferedWriter(new FileWriter(fout));
    for (String str : buf_for_sort) {
      writer.write(str + "\n");
    }
    writer.close();
  }

  public static void main(String[] args) throws IOException {
    getSentenceFocus(TRAIN_FILE_PATH, FREEBASE_ENTITY_NAME_PATH, FOCUS_TRAIN_FILE_PATH);
    getSentenceFocus(VALID_FILE_PATH, FREEBASE_ENTITY_NAME_PATH, FOCUS_VALID_FILE_PATH);
    getSentenceFocus(TEST_FILE_PATH, FREEBASE_ENTITY_NAME_PATH, FOCUS_TEST_FILE_PATH);
  }
}

