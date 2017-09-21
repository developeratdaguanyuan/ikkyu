package cfo.centralservice;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.stream.Collectors;

import javax.ws.rs.FormParam;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

import org.apache.log4j.Logger;
import org.json.JSONArray;
import org.json.JSONObject;

import com.mashape.unirest.http.HttpResponse;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import com.mashape.unirest.http.JsonNode;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

import KnowledgeGraph.KGSearch.EntitySearch;
import KnowledgeGraph.KGSearch.EntitySearch.Pair;
import nlp.tokenizer.EnglishTokenizer;
import qa.rdfDB.RDFServer;


/**
 * Root resource (exposed at "myresource" path)
 */
@Path("myresource")
public class MyResource {
  final static Logger logger = Logger.getLogger(MyResource.class);

  private String relation_server = "http://127.0.0.1:11111/relation_server";
  private String focus_server = "http://127.0.0.1:11112/focus_server";
  private String subject_server = "http://127.0.0.1:11113/subject_server";

  // KGSearch
  private static final EntitySearch entity_search = EntitySearch.getEntitySearch();

  // RDF Server
  private static final RDFServer rdf_server = new RDFServer("../../data/freebase-FB5M.txt");
  private static int cnt = 0;
  
  
  @POST
  @Path("/ask")
  @Produces(MediaType.TEXT_PLAIN)
  public String ask(@FormParam("question") String question) throws UnirestException {
    // Tokenize
    List<String> question_tokens = EnglishTokenizer.tokenizer(question.toLowerCase());
    String fine_question = question_tokens.stream().collect(Collectors.joining(" "));

    // Get Relation MIDS
    String relation_response = sendHttpPostRequest(relation_server, fine_question);
    String[] relation_response_tokens = relation_response.replaceAll("relation ranking: ", "").split(",");
    List<String> relation_mids = new ArrayList<>();
    for (int i = 0; i < 3; i++) {
      String mids = requestRedis(relation_response_tokens[i], 4);
      relation_mids.add(mids);
    }
    
    // Get focus
    String focus_response = sendHttpPostRequest(focus_server, fine_question);
    String[] focus_tokens = focus_response.replace("focus tagging: ", "").split(",");
    StringBuilder focus_builder = new StringBuilder();
    for (int i = 0; i < focus_tokens.length; i++) {
      if (Integer.parseInt(focus_tokens[i]) == 1) {
        focus_builder.append(question_tokens.get(i)).append(" ");
      }
    }
    String focus_terms = focus_builder.toString().trim();
    System.err.println(focus_terms);

    // Get Subject MIDS by retrieving in Knowledge Base 
    String mids = requestRedis(focus_terms, 2);
    if (mids == null) {
      try {
        List<String> mids_list = new ArrayList<>();
        List<Pair> pair_list = entity_search.search(focus_terms);
        for (Pair p : pair_list) {
          mids_list.add(p.mid);
        }
        if (mids_list.size() != 0) {
          mids = mids_list.stream().collect(Collectors.joining(";"));
        }
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
    
    if (mids == null) {
      // TODO: return empty
    }
    List<String> mid_list = Arrays.asList(mids.trim().split(";"));
    
    for (int i = 0; i < relation_mids.size(); i++) {
      for (int j = 0; j < mid_list.size(); j++) {
        int ret = rdf_server.isSubjectPredicateValid(mid_list.get(j), relation_mids.get(i));
        if (ret == 1) {
          System.err.println(mid_list.get(j) + "\t" + relation_mids.get(i));
          System.err.println(ret);
          System.err.println();
        }
      }
    }
    
    System.err.println("mids: " + mids);

    cnt += mid_list.size();
    System.err.println("CNT: " + cnt);
    
    return mids;
  }


  // Statistic for DEBUG
  private static int[] rel_pos = new int[101];
  private static int[] rel_pos_1 = new int[101];
  private static int[] rel_pos_2 = new int[101];

  private static int subject_mid_cnt;
  private static int subject_mid_cnt_1_c;
  private static int subject_mid_cnt_1_i;
  private static int subject_mid_cnt_2_c;
  private static int subject_mid_cnt_2_i;

  private static int object_correct_1st;
  
  private static int sub_rel_pair_hit_cnt;
  private static int correct_1st;
  private static int correct_2st;
  

  @POST
  @Path("/ask_debug")
  @Produces(MediaType.TEXT_PLAIN)
  public String ask_debug(@FormParam("question") String question,
                          @FormParam("subject_mid") String subject_mid,
                          @FormParam("predicate_mid") String predicate_mid,
                          @FormParam("object_mid") String object_mid) throws UnirestException {
    // Tokenize
    List<String> question_tokens = EnglishTokenizer.tokenizer(question.toLowerCase());
    String fine_question = question_tokens.stream().collect(Collectors.joining(" "));
    logger.info(fine_question);

    // Get Relation MIDS
    String relation_response = sendHttpPostRequest(relation_server, fine_question);
    logger.info(relation_response);
    logger.info(predicate_mid);
    String[] relation_response_tokens = relation_response.replaceAll("relation ranking: ", "").split(",");
    List<String> relation_response_id_list = new ArrayList<>();
    for (int i = 0; i < relation_response_tokens.length; i++) {
      String mid = requestRedis(relation_response_tokens[i], 4);
      relation_response_id_list.add(mid);
    }
    int rel_idx = relation_response_id_list.indexOf(predicate_mid);
    rel_idx = (rel_idx == -1 || rel_idx > 100) ? 100 : rel_idx;
    ++rel_pos[rel_idx];
    List<String> relation_mids = new ArrayList<>();
    for (int i = 0; i < 2; i++) {
      String mids = requestRedis(relation_response_tokens[i], 4);
      relation_mids.add(mids);
    }
    
    // Get focus
    String focus_response = sendHttpPostRequest(focus_server, fine_question);
    String[] focus_tokens = focus_response.replace("focus tagging: ", "").split(",");
    StringBuilder focus_builder = new StringBuilder();
    for (int i = 0; i < focus_tokens.length; i++) {
      if (Integer.parseInt(focus_tokens[i]) == 1) {
        focus_builder.append(question_tokens.get(i)).append(" ");
      }
    }
    String focus_terms = focus_builder.toString().trim();
    logger.info(focus_terms.trim());
    System.err.println(focus_terms);

    // Get Subject MIDS by retrieving Redis OR Knowledge Base 
    String mids = requestRedis(focus_terms, 2);
    if (mids == null) {
      try {
        List<String> mids_list = new ArrayList<>();
        List<Pair> pair_list = entity_search.search(focus_terms);
        for (Pair p : pair_list) {
          mids_list.add(p.mid);
        }
        if (mids_list.size() != 0) {
          mids = mids_list.stream().collect(Collectors.joining(";"));
        }
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
    if (mids == null) {
      logger.info("mids == null");
      // TODO: return empty
    }
    List<String> mid_list = Arrays.asList(mids.trim().split(";"));
    subject_mid_cnt += mid_list.size();
    if (mid_list.size() == 1) {
      int idx = mid_list.indexOf(subject_mid);
      subject_mid_cnt_1_c += (idx != -1) ? 1 : 0;
      subject_mid_cnt_1_i += (idx == -1) ? 1 : 0;
      if (idx != -1) {
        int rel_idx_t = relation_response_id_list.indexOf(predicate_mid);
        rel_idx_t = (rel_idx_t == -1 || rel_idx_t > 100) ? 100 : rel_idx_t;
        ++rel_pos_1[rel_idx_t];

        List<String> relation_name_list = new ArrayList<String>();
        for (int j = 0; j < relation_response_id_list.size(); j++) {
          if (requestRedis(relation_response_id_list.get(j), 5) == null) {
            continue;
          }
          if (rdf_server.isSubjectPredicateValid(mid_list.get(0), relation_response_id_list.get(j)) == 1) {
            relation_name_list.add(relation_response_id_list.get(j));
            if (mid_list.get(0).equals(subject_mid) && relation_response_id_list.get(j).equals(predicate_mid)) {
              if (relation_name_list.size() == 1) {
                correct_1st++;
              }
              logger.info("question relation lists: " + fine_question + "," + relation_name_list.stream().collect(Collectors.joining(",")));
              String object_mids = rdf_server.getObject(mid_list.get(0), relation_response_id_list.get(j));
              if (Arrays.asList(object_mids.split(";")).contains(object_mid)) {
                object_correct_1st++;
              }
              logger.info("object mids: " + object_mids);
              
              break;
            }
          }
        }
      }
    } else if (mid_list.size() > 1) {
      int idx = mid_list.indexOf(subject_mid);
      subject_mid_cnt_2_c += (idx != -1) ? 1 : 0;
      subject_mid_cnt_2_i += (idx == -1) ? 1 : 0;
      if (idx != -1) {
        // relation
        int rel_idx_t = relation_response_id_list.indexOf(predicate_mid);
        rel_idx_t = (rel_idx_t == -1 || rel_idx_t > 100) ? 100 : rel_idx_t;
        ++rel_pos_2[rel_idx_t];
        // subject
        Map<String, String> id_mid_mapper = new HashMap<>();
        StringBuilder sids_builder = new StringBuilder();
        for (String sid: mid_list) {
          if (sids_builder.length() != 0) {
            sids_builder.append(",");
          }
          String id = requestRedis(sid, 0);
          id_mid_mapper.put(id, sid);
          sids_builder.append(id);
        }
        logger.info("[fine_question] " + fine_question + " [subject_ids] " + sids_builder.toString());
        Map<String, Object> requests = new HashMap<>();
        requests.put("question", fine_question);
        requests.put("subject_ids", sids_builder.toString());
        String subject_response = sendHttpPostRequest(subject_server, requests);
        String[] subject_tokens = subject_response.replace("subject ranking: ", "").trim().split(",");
        for (int i = 0; i < relation_response_id_list.size(); i++) {
          for (String id: subject_tokens) {
            String curr_mid = id_mid_mapper.get(id);
            if (rdf_server.isSubjectPredicateValid(curr_mid, relation_response_id_list.get(i)) == 1) {
              if (curr_mid.equals(subject_mid) && relation_response_id_list.get(i).equals(predicate_mid)) {
                ++correct_2st;
              }
              break;
            }
          }
        }
      }
    }

    // Merge Relation and Subject
    List<String> sub_rel_list = new ArrayList<>();
    for (int i = 0; i < relation_mids.size(); i++) {
      for (int j = 0; j < mid_list.size(); j++) {
        int ret = rdf_server.isSubjectPredicateValid(mid_list.get(j), relation_mids.get(i));
        if (ret == 1) {
          String object_mid_t = rdf_server.getObject(mid_list.get(j), relation_mids.get(i));
          sub_rel_list.add(mid_list.get(j) + "\t" + relation_mids.get(i) + "\t" + object_mid);
          if (mid_list.get(j).equals(subject_mid) && relation_mids.get(i).equals(predicate_mid)) {
            sub_rel_pair_hit_cnt++;
          }
        }
      }
    }
 
    logger.info("subject_mid_cnt: " + subject_mid_cnt);
    logger.info("subject_mid_cnt_1_c: " + subject_mid_cnt_1_c);
    logger.info("subject_mid_cnt_1_i: " + subject_mid_cnt_1_i);
    logger.info("subject_mid_cnt_2_c: " + subject_mid_cnt_2_c);
    logger.info("subject_mid_cnt_2_i: " + subject_mid_cnt_2_i);
    logger.info("sub_rel_pair_hit_cnt: " + sub_rel_pair_hit_cnt);
    logger.info("correct_1st: " + correct_1st);
    logger.info("correct_2st: " + correct_2st);
    logger.info("object_correct_1st: " + object_correct_1st);
    logger.info("Relation ranking: " + Arrays.toString(rel_pos));
    logger.info("1st Relation ranking: " + Arrays.toString(rel_pos_1));
    logger.info("2nd Relation ranking: " + Arrays.toString(rel_pos_2));
   
    return mids;
  }

  private String sendHttpPostRequest(String url, String question) {
    String result = null;
    try {
      HttpResponse<String> response = Unirest.post(url).field("question", question).asString();
      System.err.println(response.getStatus());
      System.err.println(response.getBody().toString());
      result = response.getBody().toString().trim();
    } catch (UnirestException e) {
      System.err.println("Caught IOException: " + e.getMessage());
    }
    return result;
  }

  private String sendHttpPostRequest(String url, Map<String, Object> requests) {
    String result = null;
    try {
      HttpResponse<String> response = Unirest.post(url).fields(requests).asString();
      System.err.println(response.getStatus());
      System.err.println(response.getBody().toString());
      result = response.getBody().toString().trim();
    } catch (UnirestException e) {
      System.err.println("Caught IOException: " + e.getMessage());
    }
    return result;
  }

  private List<String> sendHttpGetRequest(String url, String question) {
    List<String> result = new ArrayList<>();
    try {
      HttpResponse<JsonNode> response = Unirest.get(url).routeParam("param", question).asJson();
      JsonNode json_body = response.getBody();
      JSONObject json_object = json_body.getObject();
      JSONObject json_header = (JSONObject) json_object.get("responseHeader");
      if ((int) json_header.get("status") != 0) {
        System.err.println((int) json_header.get("status"));
        return null;
      }
      JSONObject json_response = (JSONObject) json_object.get("response");
      JSONArray doc_arr = (JSONArray) json_response.get("docs");
      for (int i = 0; i < doc_arr.length(); i++) {
        JSONObject obj = (JSONObject) doc_arr.get(i);
        JSONArray mid_arr = (JSONArray) obj.get("mid");
        if (mid_arr.length() > 0) {
          result.add((String) mid_arr.get(0));
        }
      }
    } catch (UnirestException e) {
      System.err.println("Caught IOException: " + e.getMessage());
    }

    return result;
  }

  private String requestRedis(String query, int db_id) {
    String result = null;
    JedisPool pool = new JedisPool(new JedisPoolConfig(), "localhost");
    Jedis jedis = pool.getResource();
    jedis.select(db_id);
    if (jedis.exists(query)) {
      result = jedis.get(query);
    }
    if (jedis != null) {
      jedis.close();
    }
    pool.close();
    pool.destroy();

    return result;
  }
}
