package qa.rdfDB;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.lang.StringBuilder;

import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.Property;
import org.apache.jena.rdf.model.Resource;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.rdf.model.StmtIterator;


/**
 * RDF server
 */
public class RDFServer {
  private static final String USELESS_PATTERN = "www.freebase.com";
  
  // input
  private static final String FREEBASE_5M_PATH = "../../data/freebase-FB5M.txt";

  private static Map<String, Resource> resource_map = new HashMap<String, Resource>();
  private static Map<String, Property> property_map = new HashMap<String, Property>();
  
  private static Model model = null;
  
  private static String normalize(String text) {
    if (text.substring(0, USELESS_PATTERN.length()).equals(USELESS_PATTERN)) {
      text = text.replaceFirst(USELESS_PATTERN, "");
    }
    return text;
  }
  
  public static void buildModel(String data_path) throws IOException {
    if (model != null && !resource_map.isEmpty() && !property_map.isEmpty()) {
      return;
    }
    // Create an empty Model
    model = ModelFactory.createDefaultModel();
    
    //File fin = new File(FREEBASE_5M_PATH);
    File fin = new File(data_path);
    BufferedReader reader = new BufferedReader(new FileReader(fin));

    String line;
    while ((line = reader.readLine()) != null) {
      String[] tokens = line.trim().split("\t");
      String subject = normalize(tokens[0]);
      String predicate = normalize(tokens[1]);
      
      Resource sub_resource = null;
      if (resource_map.containsKey(subject)) {
        sub_resource = resource_map.get(subject);
      } else {
        sub_resource = model.createResource(subject);
        resource_map.put(subject, sub_resource);
      }
      
      Property prd_property = null;
      if (property_map.containsKey(predicate)) {
        prd_property = property_map.get(predicate);
      } else {
        prd_property = model.createProperty(predicate);
        property_map.put(predicate, prd_property);
      }
      
      String[] objects = tokens[2].split(" ");
      for (String obj : objects) {
        String object = normalize(obj);
        Resource obj_resource = null;
        if (resource_map.containsKey(object)) {
          obj_resource = resource_map.get(object);
        } else {
          obj_resource = model.createResource(object);
          resource_map.put(object, obj_resource);
        }
        sub_resource.addProperty(prd_property, obj_resource);
      }
    }
    
    File file = new File("/tmp/ntriples");
    FileOutputStream fop = new FileOutputStream(file);
    model.write(fop, "N-TRIPLE");
    
    System.out.println(model.size());
    reader.close();
  }

  public static int isSubjectPredicateValid(String sub_mid, String prd_mid) {
    Resource subject = model.createResource(sub_mid);
    Property predicate = model.createProperty(prd_mid);
    StmtIterator iter = model.listStatements(subject, predicate, (Resource)null);
    return iter.hasNext() ? 1 : 0;
  }
  
  public static String getObject(String sub_mid, String prd_mid) {
    //String object = null;
    Resource subject = model.createResource(sub_mid);
    Property predicate = model.createProperty(prd_mid);
    StmtIterator iter = model.listStatements(subject, predicate, (Resource)null);

    StringBuilder builder = new StringBuilder();
    while (iter.hasNext()) {
      Statement stmt = iter.nextStatement();
      builder.append(stmt.getObject().toString()).append(";");
    }
    return builder.toString();
  }
  
  public RDFServer(String data_path) {
    try {
      System.err.println("Hello world");
      buildModel(data_path);
    } catch (IOException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }
  
  public static void main(String[] args) throws IOException {
    buildModel(FREEBASE_5M_PATH);
    System.out.println(getObject("/m/03_7vl", "/people/person/profession"));
    System.out.println(isSubjectPredicateValid("/m/02vn_x3", "/people/person/nationality"));
    System.out.println(isSubjectPredicateValid("/m/09c7w0", "/people/person/nationality"));
    System.out.println(isSubjectPredicateValid("/people/person/nationality", "/people/person/nationality"));
    System.out.println(isSubjectPredicateValid("/m/09c7w0", "/people/person/father"));
    System.out.println(getObject("/m/01jp8ww", "/music/album/genre"));
  }
}
