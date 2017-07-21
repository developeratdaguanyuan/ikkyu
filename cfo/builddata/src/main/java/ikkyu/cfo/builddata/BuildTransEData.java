package ikkyu.cfo.builddata;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/*
 * @function: generate triples, entity-id map, relation-id map
 * 
 * @date: 2016-09-08
 */
public class BuildTransEData {
	private static final String USELESS_PATTERN = "www.freebase.com";

	// Input Paths
	private static final String FREEBASE_5M_PATH = "../../data/freebase-FB5M.txt";
	// Output Paths
	private static final String FREEBASE_TRIPLES_PATH = "../transE/data/FB5M-triples.txt";
	private static final String FREEBASE_ENTITY_ID_PATH = "../transE/data/FB5M-entity-id.txt";
	private static final String FREEBASE_RELATION_ID_PATH = "../transE/data/FB5M-relation-id.txt";

	private static HashMap<String, Integer> entityMap = new HashMap<String, Integer>();
	private static HashMap<String, Integer> predicateMap = new HashMap<String, Integer>();

	public static void getTriplesDigit(String iPath, String oPath_1, String oPath_2, String oPath_3)
			throws IOException {
		{
			File fin = new File(iPath);
			BufferedReader reader = new BufferedReader(new FileReader(fin));
			File fout = new File(oPath_1);
			BufferedWriter writer = new BufferedWriter(new FileWriter(fout));

			String line;
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.trim().split("\t");
				String subject = tokens[0];
				String predicate = tokens[1];
				int subjectID = -1;
				if (entityMap.containsKey(subject)) {
					subjectID = entityMap.get(subject);
				} else {
					subjectID = entityMap.size();
					entityMap.put(subject, subjectID);
				}
				int predicateID = -1;
				if (predicateMap.containsKey(predicate)) {
					predicateID = predicateMap.get(predicate);
				} else {
					predicateID = predicateMap.size();
					predicateMap.put(predicate, predicateID);
				}

				String[] objects = tokens[2].split(" ");
				for (int i = 0; i < objects.length; i++) {
					String object = objects[i];
					int objectID = -1;
					if (entityMap.containsKey(object)) {
						objectID = entityMap.get(object);
					} else {
						objectID = entityMap.size();
						entityMap.put(object, objectID);
					}
					writer.write(subjectID + "\t" + predicateID + "\t" + objectID + "\n");
				}
			}
			writer.close();
			reader.close();
		}
		{
			File file = new File(oPath_2);
			BufferedWriter writer = new BufferedWriter(new FileWriter(file));
			Set<Map.Entry<String, Integer>> entrySet = entityMap.entrySet();
			for (Entry entry : entrySet) {
				String key = entry.getKey().toString();
				if (key.substring(0, USELESS_PATTERN.length()).equals(USELESS_PATTERN)) {
					key = key.replaceFirst(USELESS_PATTERN, "");
				}
				writer.write(key + "\t" + entry.getValue() + "\n");
			}
			writer.close();
		}
		{
			File file = new File(oPath_3);
			BufferedWriter writer = new BufferedWriter(new FileWriter(file));
			Set<Map.Entry<String, Integer>> entrySet = predicateMap.entrySet();
			for (Entry entry : entrySet) {
				String key = entry.getKey().toString();
				if (key.substring(0, USELESS_PATTERN.length()).equals(USELESS_PATTERN)) {
					key = key.replaceFirst(USELESS_PATTERN, "");
				}
				writer.write(key + "\t" + entry.getValue() + "\n");
			}
			writer.close();
		}
		System.out.println("entityMap size: " + entityMap.size());
		System.out.println("predicateMap size: " + predicateMap.size());
	}

	public static void main(String[] args) throws IOException {
		BuildTransEData.getTriplesDigit(FREEBASE_5M_PATH, FREEBASE_TRIPLES_PATH, FREEBASE_ENTITY_ID_PATH,
				FREEBASE_RELATION_ID_PATH);
	}
}
