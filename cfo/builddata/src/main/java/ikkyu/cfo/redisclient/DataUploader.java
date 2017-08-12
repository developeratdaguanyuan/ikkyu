package ikkyu.cfo.redisclient;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import edu.stanford.nlp.util.StringUtils;
import nlp.tokenizer.EnglishTokenizer;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

public class DataUploader {
  public static void UploadEntityMID2NameOrID(String path, int db_id) {
    Jedis jedis = null;
    JedisPool pool = new JedisPool(new JedisPoolConfig(), "localhost");
    try {
      jedis = pool.getResource();
      jedis.select(db_id);
      String line;
      File fin = new File(path);
      BufferedReader reader = new BufferedReader(new FileReader(fin));
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.trim().split("\t");
        if (tokens.length < 2) {
          continue;
        }
        String subject_mid = tokens[0].trim().replace("www.freebase.com", "");
        StringBuilder builder = new StringBuilder();
        for (int i = 1; i < tokens.length; i++) {
          builder.append(tokens[i]).append(" ");
        }
        String value = builder.toString().trim();
        if (value.length() > 2 && value.charAt(0) == '\"' && value.charAt(value.length() - 1) == '\"') {
          value = value.substring(1, value.length() - 1);
        }
        jedis.set(subject_mid, value);
      }
      reader.close();
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } finally {
      if (jedis != null) {
        jedis.close();
      }
    }
    pool.destroy();
  }

  public static void UploadEntityMID2Embeddings(String path, int db_id) {
    Jedis jedis = null;
    JedisPool pool = new JedisPool(new JedisPoolConfig(), "localhost");
    try {
      jedis = pool.getResource();
      jedis.select(db_id);
      String line;
      File fin = new File(path);
      BufferedReader reader = new BufferedReader(new FileReader(fin));
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.trim().split(":");
        jedis.set(tokens[0], tokens[1]);
      }
      reader.close();
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } finally {
      if (jedis != null) {
        jedis.close();
      }
    }
    pool.destroy();
  }

  public static void UploadEntityname2MIDs(String path, int db_id) {
    Jedis jedis = null;
    JedisPool pool = new JedisPool(new JedisPoolConfig(), "localhost");
    try {
      jedis = pool.getResource();
      jedis.select(db_id);
      String line;
      File fin = new File(path);
      BufferedReader reader = new BufferedReader(new FileReader(fin));
      while ((line = reader.readLine()) != null) {
        String[] tokens = line.trim().split("\t");
        if (tokens.length < 2) {
          continue;
        }
        StringBuilder builder = new StringBuilder();
        for (int i = 1; i < tokens.length; i++) {
          builder.append(tokens[i]).append(" ");
        }
        String name = builder.toString().trim();
        if (name.length() > 2 && name.charAt(0) == '\"' && name.charAt(name.length() - 1) == '\"') {
          name = name.substring(1, name.length() - 1);
        }
        List<String> name_tokens = EnglishTokenizer.tokenizer(name.toLowerCase());
        name = name_tokens.stream().collect(Collectors.joining(" "));

        String mid = tokens[0].replace("www.freebase.com", "");
        if (jedis.exists(name)) {
          Set<String> set = new HashSet<String>(Arrays.asList(jedis.get(name).split(";")));
          if (!set.contains(mid)) {
            mid = jedis.get(name) + ";" + mid;
            jedis.set(name, mid);
          }
        } else {
          jedis.set(name, mid);
        }
      }
      reader.close();
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    } finally {
      if (jedis != null) {
        jedis.close();
      }
    }
    pool.destroy();
  }


  public final static void main(String[] args) {
    UploadEntityMID2NameOrID("../data/transE/FB5M-entity-id.txt", 0);
    UploadEntityMID2NameOrID("../../data/entity_name_map.txt", 1);
    UploadEntityMID2Embeddings("../data/transE/entity_embeddings.txt", 2);
    UploadEntityname2MIDs("../../data/entity_name_map.txt", 3);
  }
}
