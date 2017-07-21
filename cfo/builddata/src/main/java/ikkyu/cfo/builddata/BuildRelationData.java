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

public class BuildRelationData {
	// Input files
	private static final String FREEBASE_RELATION_NAME_PATH = "../../data/FB5M-relation-id-tiny.txt";
	private static final String TRAIN_FILE_PATH = "../../data/SimpleQuestions_v2/annotated_fb_data_train.txt";
	private static final String VALID_FILE_PATH = "../../data/SimpleQuestions_v2/annotated_fb_data_valid.txt";
	private static final String TEST_FILE_PATH = "../../data/SimpleQuestions_v2/annotated_fb_data_test.txt";

	// Output files
	private static final String RELATION_TRAIN_FILE_PATH = "../relation/data/relation_train.txt";
	private static final String RELATION_VALID_FILE_PATH = "../relation/data/relation_valid.txt";
	private static final String RELATION_TEST_FILE_PATH = "../relation/data/relation_test.txt";

	private static void buildRelationData(String question_path, String relation_name_path, String oPath)
			throws IOException {
		// Load relation-id map
		File fin = new File(relation_name_path);
		BufferedReader reader = new BufferedReader(new FileReader(fin));
		Map<String, String> relation_id_map = new HashMap<String, String>();
		String line;
		while ((line = reader.readLine()) != null) {
			String[] tokens = line.trim().split("\t");
			String relation_name = tokens[0].trim();
			String relation_id = tokens[1].trim();
			if (!relation_id_map.containsKey(relation_name)) {
				relation_id_map.put(relation_name, relation_id);
			}
		}
		reader.close();

		File fout = new File(oPath);
		BufferedWriter writer = new BufferedWriter(new FileWriter(fout));
		fin = new File(question_path);
		reader = new BufferedReader(new FileReader(fin));
		while ((line = reader.readLine()) != null) {
			String[] tokens = line.trim().split("\t");

			String relation_name = tokens[1].trim().replace("www.freebase.com", "");
			String relation_id = relation_id_map.get(relation_name);

			String question = tokens[3].trim();
			List<String> question_tokens = EnglishTokenizer.tokenizer(question.toLowerCase());
			writer.write(question_tokens.stream().collect(Collectors.joining(" ")) + "\t" + relation_id + "\n");
		}
		reader.close();
		writer.close();
	}

	public static void main(String[] args) throws IOException {
		buildRelationData(TRAIN_FILE_PATH, FREEBASE_RELATION_NAME_PATH, RELATION_TRAIN_FILE_PATH);
		buildRelationData(VALID_FILE_PATH, FREEBASE_RELATION_NAME_PATH, RELATION_VALID_FILE_PATH);
		buildRelationData(TEST_FILE_PATH, FREEBASE_RELATION_NAME_PATH, RELATION_TEST_FILE_PATH);
	}

}
