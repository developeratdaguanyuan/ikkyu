package nlp.tokenizer;

import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.PTBTokenizer;

public class EnglishTokenizer {
  public static List<String> tokenizer(String text) {
    PTBTokenizer<CoreLabel> ptbt =
        new PTBTokenizer<CoreLabel>(new StringReader(text), new CoreLabelTokenFactory(), "normalizeParentheses=false,normalizeOtherBrackets=false,latexQuotes=false,asciiQuotes=true,escapeForwardSlashAsterisk=false");
    List<String> tokens = new LinkedList<String>();
    while (ptbt.hasNext()) {
      CoreLabel label = ptbt.next();
      tokens.add(label.value());
    }
    return tokens;
  }

  public static void main(String[] args) throws IOException {
    /*
    List<String> tokens = EnglishTokenizer.tokenizer("(\'Hello world......\')octinoxate/titanium/zinc");
    for (String token : tokens) {
      System.out.println(token);
    }*/
    /*
    List<String> data_list = new ArrayList<>();
    String[] categories = {"comp", "politics", "rec", "religion"};
    for (String category : categories) {
      try (Stream<Path> paths = Files.walk(Paths.get(
          "/home/dzhou/Downloads/20news-bydate/20news-bydate-test/Untitled Folder/" + category))) {
        paths.forEach(filePath -> {
          if (Files.isRegularFile(filePath)) {
            try {
              List<String> tokens = new ArrayList<>();
              List<String> lines = Files.readAllLines(filePath);
              for (String str : lines) {
                tokens.addAll(EnglishTokenizer.tokenizer(str.trim().toLowerCase()));
              }
              data_list.add(tokens.stream().collect(Collectors.joining(" ")) + "\t" + category);
            } catch (IOException e) {
              e.printStackTrace();
            }
          }
        });
      } catch (Exception e) {
        System.out.println(e);
      }
    }
    Collections.shuffle(data_list);
    
    FileWriter fw = new FileWriter("/home/dzhou/Downloads/20news-bydate/20news-bydate-test/Untitled Folder/test.txt");
    for (String data : data_list) {
      fw.write(data.trim() + "\n");
    }
    fw.close();
    */
    
    List<String> tokens = EnglishTokenizer.tokenizer("Who created la bohème, act iv: in un coupé??".toLowerCase());
    for (String token : tokens) {
      System.out.println(token);
    }
    
  }
}
