package KnowledgeGraph.KGSearch;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * entity search by kgsearch API
 *
 */
public class EntitySearch {
  private static String password = "";
  private volatile static EntitySearch entity_search = null;

  private EntitySearch() {
    File fin = new File("./password");
    BufferedReader reader;
    try {
      reader = new BufferedReader(new FileReader(fin));
      password = reader.readLine();
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

  public static EntitySearch getEntitySearch() {
    if (entity_search == null) {
      synchronized (EntitySearch.class) {
        if (entity_search == null) {
          entity_search = new EntitySearch();
        }
      }
    }
    return entity_search;
  }

  public class Pair {
    public String mid;
    public String name;
    public double score;

    public String toString() {
      return mid + "\t" + name + "\t" + score;
    }
  }

  public List<Pair> search(String query) throws Exception {
    if (query == null || query.length() == 0) {
      return null;
    }

    List<Pair> rank_list = new ArrayList<Pair>();
    String urlToRead = "https://kgsearch.googleapis.com/v1/entities:search?query="
        + URLEncoder.encode(query, "UTF-8") + "&key=" + password + "&limit=10&indent=True";
    HttpURLConnection conn = null;
    for (int i = 0; i < 2; i++) {
      URL url = new URL(urlToRead);
      conn = (HttpURLConnection) url.openConnection();
      conn.setRequestMethod("GET");
      if (conn.getResponseCode() == 200) {
        TimeUnit.MILLISECONDS.sleep(200);
        break;
      }
      TimeUnit.MILLISECONDS.sleep(200);
    }

    if (conn.getResponseCode() >= 300) {
      throw new Exception(
          "HTTP Request in KGSearch is not success, Response code is " + conn.getResponseCode());
    }

    BufferedReader rd = new BufferedReader(new InputStreamReader(conn.getInputStream()));
    StringBuilder result = new StringBuilder();
    String line;
    while ((line = rd.readLine()) != null) {
      result.append(line);
    }
    rd.close();

    String json = result.toString();
    JSONObject jsonObj = new JSONObject(json);
    JSONArray jsonArr = jsonObj.getJSONArray("itemListElement");
    for (int i = 0; i < jsonArr.length(); i++) {
      Pair pair = new Pair();

      JSONObject obj = (JSONObject) jsonArr.get(i);
      JSONObject res = (JSONObject) obj.get("result");
      if (res != null) {
        String id_kg = (String) res.get("@id");
        if (id_kg != null) {
          String[] tokens = id_kg.split(":");
          if (tokens.length == 2) {
            pair.mid = tokens[1].trim();
          } else {
            pair.mid = tokens[tokens.length - 1].trim();
          }
        }
        if (res.has("name")) {
          pair.name = (String) res.get("name");
        } else {
          pair.name = null;
        }
      }
      Double score = obj.getDouble("resultScore");
      if (score != null) {
        pair.score = score.doubleValue();
      }

      rank_list.add(pair);
    }

    return rank_list;
  }

  public static void main(String[] args) throws Exception {
    EntitySearch entity_search = getEntitySearch();
    List<Pair> rank_list = entity_search.search("taylor swift");
    for (Pair p : rank_list) {
      System.out.println(p.toString());
    }
  }
}
