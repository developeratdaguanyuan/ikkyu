package nlp.tokenizer;

import java.io.IOException;
import java.io.StringReader;
import java.util.LinkedList;
import java.util.List;

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
    List<String> tokens = EnglishTokenizer.tokenizer("Who created la bohème, act iv: in un coupé??".toLowerCase());
    for (String token : tokens) {
      System.out.println(token);
    }
    
  }
}
