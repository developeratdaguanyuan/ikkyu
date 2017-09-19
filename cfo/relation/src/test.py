import commands

f = open('../../data/relation/relation_test.txt')
lines = f.readlines()
f.close()
for line in lines:
  try:
    tokens = line.strip().split('\t')
    question = 'question=' + tokens[0]
    tag = int(tokens[1])
    output = commands.getstatusoutput('curl -d \"' + question + '\" http://127.0.0.1:11111/relation_server')
    ranking = output[1].split('relation ranking: ')[1].strip().split(',')
    ranking = [int(r) for r in ranking]
    ranking = ranking.index(tag) if tag in ranking else 100
    print str(ranking) + '\t' + str(tag) + '\t' + tokens[0]
  except:
    pass
