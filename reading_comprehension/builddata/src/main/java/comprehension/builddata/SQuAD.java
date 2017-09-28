package comprehension.builddata;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

import org.json.JSONArray;
import org.json.JSONObject;

import nlp.tokenizer.EnglishTokenizer;

public class SQuAD {
  // Input files
  private static String TRAIN_FILE_PATH = "../../data/SQuAD/train-v1.1.json";
  private static String VALID_FILE_PATH = "../../data/SQuAD/dev-v1.1.json";
  
  // Output files
  private static String TRAIN_FILE_OUTPUT_PATH = "../data/SQuAD/train-v1.1.txt";
  private static String VALID_FILE_OUTPUT_PATH = "../data/SQuAD/dev-v1.1.txt";

  private static void parseJson(String inpath, String outpath) throws IOException {
    FileWriter fw = new FileWriter(outpath);

    String json = new String(Files.readAllBytes(Paths.get(inpath)));
    JSONObject jsonObj = new JSONObject(json);
    JSONArray jsonArr = jsonObj.getJSONArray("data");
    for (int i = 0; i < jsonArr.length(); i++) {
      JSONObject para = (JSONObject) jsonArr.get(i);
      JSONArray sample = (JSONArray) para.getJSONArray("paragraphs");
      for (int j = 0; j < sample.length(); j++) {
        JSONObject context = (JSONObject) sample.get(j);
        String cntxt = context.getString("context").trim().replaceAll("\n", " ");
        JSONArray qas = context.getJSONArray("qas");
        for (int k = 0; k < qas.length(); k++) {
          JSONObject pair = (JSONObject) qas.get(k);
          String question = ((String) pair.get("question")).trim();
          String id = ((String) pair.get("id")).trim();
          JSONArray answer_array = (JSONArray) pair.get("answers");
          for (int l = 0; l < answer_array.length(); l++) {
            JSONObject ans = (JSONObject) answer_array.get(l);
            Integer answer_start = ans.getInt("answer_start");
            String text = ((String) ans.get("text")).trim();
            List<String> part_a = EnglishTokenizer.tokenizer(cntxt.substring(0, answer_start).toLowerCase());
            List<String> whole = EnglishTokenizer.tokenizer(cntxt.toLowerCase());
            String cntxt_tokenized = whole.stream().collect(Collectors.joining(" "));
            answer_start = part_a.stream().collect(Collectors.joining(" ")).length() + 1;
            String text_tokenized = EnglishTokenizer.tokenizer(text.toLowerCase()).stream().collect(Collectors.joining(" "));
            String question_tokenized = EnglishTokenizer.tokenizer(question.toLowerCase()).stream().collect(Collectors.joining(" "));
            fw.write(cntxt_tokenized + "\t" + question_tokenized + "\t" + id + "\t" + answer_start.toString() + "\t" + text_tokenized + "\n");
            assert(cntxt_tokenized.substring(answer_start, answer_start + text_tokenized.length()).equals(text_tokenized));
          }
        }
      }
    }
    
    fw.close();
  }
  
  public static void main(String[] args) throws IOException {
    parseJson(TRAIN_FILE_PATH, TRAIN_FILE_OUTPUT_PATH);
    parseJson(VALID_FILE_PATH, VALID_FILE_OUTPUT_PATH);
  }
}
