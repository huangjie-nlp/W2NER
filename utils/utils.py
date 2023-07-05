
import json

def convert_index_to_text(index, type):
    index2string = '-'.join([str(i) for i in index])
    ner_text = index2string + '-#-{}'.format(type)
    return ner_text

def decode(outputs, length, sentence, id2label):
    # id2label = json.load(open(schema_fn, "r", encoding="utf-8"))[1]
    for index, (logits, leng) in enumerate(zip(outputs, length)):
        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(leng):
            for j in range(i + 1, leng):
                if logits[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)

        for i in range(leng):
            for j in range(i, leng):
                if logits[j, i] > 1:
                    ht_type_dict[(i, j)] = logits[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)
        predicts = []
        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())

            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        for head in head_dict:
            find_entity(head, [], head_dict[head])
        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])
        result = []
        for pred in predicts:
            res = pred.split("-#-")
            entity = res[0]
            entity_type = res[1]
            e = ""
            for char_id in entity.split("-"):
                e += sentence[int(char_id)]
            st, et = entity.split("-") 
            result.append(id2label[str(entity_type)] + "/" + st + "/" + et + "#" + e)

        return list(predicts), result

def parser_indx_to_text(li, sentence):
    result = []
    for i in li:
        temp = []
        e_type = ''
        for inx in i["index"]:
            temp.append(sentence[inx])
            e_type = i['type']
        if temp == [] or e_type == '':
            continue
        e = ''.join(temp)
        entity = e_type + '/' + str(i["index"][0]) + '/' +str(i["index"][-1]) + '#' + e
        result.append(entity)
    return result
