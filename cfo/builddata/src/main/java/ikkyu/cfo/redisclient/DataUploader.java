package ikkyu.cfo.redisclient;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

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
        String subject_mid = tokens[0].trim().replace("www.freebase.com", "");
        jedis.set(subject_mid, tokens[1]);
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

  public final static void main(String[] args) {
    UploadEntityMID2NameOrID("../data/transE/FB5M-entity-id.txt", 0);
    UploadEntityMID2NameOrID("../../data/entity_name_map.txt", 1);
    UploadEntityMID2Embeddings("../data/transE/entity_embeddings.txt", 2);
  }
}
