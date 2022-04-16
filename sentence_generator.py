#!/usr/bin/env python
# coding: utf-8

import numpy
import random
import json
import sys
import re


def populate_libraries(np_fp, vb_fp, pp_fp, pth_fp, emd_fp, verb_tense_fp):
    """Open all vocabulary files and load vocabulary to memory."""

    nps = []
    verbs = []
    pps = []
    parenths = []
    embeds = []

    with open(np_fp) as f:
        for line in f:
            line = line[:-1]
            np = line.split(':=')[0]
            plurality = line.split(':=')[1]
            nps.append({'words': np, 'plurality': plurality})

    with open(vb_fp) as f:
        for line in f:
            line = line[:-1]
            line = line.split(':=')
            verbs.append(line)

    with open(pp_fp) as f:
        for line in f:
            pps.append(line[:-1])

    with open(pth_fp) as f:
        for line in f:
            line = line[:-1]
            parenthetical = line.split(':=')[0]
            parenthetical_type = line.split(':=')[1]
            parenths.append([parenthetical, parenthetical_type])

    with open(emd_fp) as f:
        for line in f:
            line = line[:-1]
            emb = line.split(':=')[0]
            emb_type = line.split(':=')[1]
            embeds.append([emb, emb_type])

    with open(verb_tense_fp) as f:
        all_verbs = json.load(f)

    for verb in verbs:
        #print(verb[0].split()[0])
        if all_verbs.get(verb[0].split()[0]):
            continue
        else:
            print(verb)

    return nps, verbs, pps, parenths, embeds, all_verbs


def generate_wordsets(probs, given_np=None, np2=None, given_verb=None):
    """Iteratively generate word sets. (inefficient, but faster than it was)"""

    new_word_set = {'noun_phrases': None,
                    'verb': None,
                    'verb_argument': None,
                    'prepositional_phrase': None,
                    'parenthetical': None,
                    'quotative_embedding': None,
                    'auxiliary_embedding': False
                    }
    
    ####
    # Run until wordset has a unique set of noun phrases
    while new_word_set['noun_phrases'] in np_sets or not new_word_set['noun_phrases']:
        if given_np:
            np1 = given_np
            if np2:
                while np2 in distributed:
                    #print(np2, end=' ')
                    np2 = numpy.random.choice(nps, size=1, p=probs)[0]
                    #print(np2)
            else:    
                np2 = numpy.random.choice(nps, size=1, p=probs)[0]
                while np1 == np2:
                    np2 = numpy.random.choice(nps, size=1, p=probs)[0]
        else:
            np1, np2 = numpy.random.choice(nps, size=2, replace=False, p=probs)

        if nps.index(np1) < nps.index(np2):
            new_word_set['noun_phrases'] = [np1, np2]
        else:
            new_word_set['noun_phrases'] = [np2, np1]
    ####

    np_sets.append(new_word_set['noun_phrases'])

    if given_verb:
        verb_set = given_verb
        if len(verb_set) is 1:
            new_word_set['verb'] = verb_set[0]
        else:
            new_word_set['verb'] = verb_set[0]
            new_word_set['verb_argument'] = verb_set[1]
    else:
        verb_set = random.sample(verbs, k=1)[0]
        if len(verb_set) is 1:
            new_word_set['verb'] = verb_set[0]
        else:
            new_word_set['verb'] = verb_set[0]
            new_word_set['verb_argument'] = verb_set[1]

    if random.random() < PP_PROBABILITY:
        pp = random.sample(pps, k=1)[0]
        new_word_set['prepositional_phrase'] = pp

    if random.random() < PTH_PROBABILITY:
        parenthetical_elements = random.sample(parenths, k=1)[0]
        parenthetical = parenthetical_elements[0]
        parenthetical_type = parenthetical_elements[1]
        new_word_set['parenthetical'] = {
            'words': parenthetical, 'type': parenthetical_type}

    if random.random() < EMD_PROBABILITY:
        quotative_embedding = random.sample(embeds, k=1)[0]
        new_word_set['quotative_embedding'] = quotative_embedding

    if random.random() < AUXEMD_PROBABILITY:
        new_word_set['auxiliary_embedding'] = True

    wordsets.append(new_word_set)


def convert_sentences_to_strings(item):
    """Convert sentence element list into a string, with proper capitalization and punctuation."""

    for sentence in item:

        sentence_string = ''
        for i in range(len(sentence)):
            phrase = sentence[i]
            sentence_string += ' ' + phrase

        sentence_string = re.sub(r'^ ', r'', sentence_string)
        sentence_string = re.sub(r'^, ', r'', sentence_string)
        sentence_string = re.sub(r' ,$', r'', sentence_string)
        sentence_string = re.sub(r' ,', r',', sentence_string)
        sentence_string = re.sub(r'$', r'.', sentence_string)

        sentence_string = sentence_string[:1].upper() + sentence_string[1:]

        yield sentence_string


# Sentence changers

def make_acc(np):
    """Ensure NP is in accusative case."""

    np_parts = np.split()
    remainder = ''
    for word in np_parts[1:]:
        remainder = f'{remainder} {word}'

    if np_parts[0] == 'she':
        np = f'her{remainder}'
    elif np_parts[0] == 'he':
        np = f'him{remainder}'
    elif np_parts[0] == 'we':
        np = f'us{remainder}'
    elif np_parts[0] == 'I':
        np = f'me{remainder}'
    elif np_parts[0] == 'they':
        np = f'them{remainder}'

    return np


def make_nom(np):
    """Ensure NP is in nominative case."""

    if np in np_list:
        return np
    else:
        np_parts = np.split()
        remainder = ''
        for word in np_parts[1:]:
            remainder = f'{remainder} {word}'

        if np_parts[0] == 'her':
            np = f'she{remainder}'
        elif np_parts[0] == 'him':
            np = f'he{remainder}'
        elif np_parts[0] == 'us':
            np = f'we{remainder}'

        return np


def assign_noun_phrase_agreement(wordset, np1=None, np2=None):
    """Ensure past 'be' form agrees with number."""

    if not np1:
        np1 = wordset['noun_phrases'][0]['words']
    if not np2:
        np2 = wordset['noun_phrases'][1]['words']
    np1_pbe = 'was'
    if wordset['noun_phrases'][0]['plurality'] == '1':
        np1_pbe = 'were'
    np2_pbe = 'was'
    if wordset['noun_phrases'][1]['plurality'] == '1':
        np2_pbe = 'were'
    return np1, np2, np1_pbe, np2_pbe


def assign_verb_conjugation(wordset, np1_pbe, np2_pbe):
    verb = wordset['verb']
    verb_parts = verb.split()
    remainder = ''
    for word in verb_parts[1:]:
        remainder = f'{remainder} {word}'
    
    if wordset['auxiliary_embedding']:

        if all_verbs.get(verb_parts[0]):
            verb = all_verbs.get(verb_parts[0])[0] + remainder
            ppart_verb = all_verbs.get(verb_parts[0])[3] + remainder
        else:
            print(verb, '*********\n')
    else:
        if all_verbs.get(verb_parts[0]):
            verb = all_verbs.get(verb_parts[0])[2] + remainder
            ppart_verb = all_verbs.get(verb_parts[0])[3] + remainder
        else:
            print(verb, '*********\n')
    return verb, ppart_verb


def add_verb_argument(wordset, item):

    for sentence_i, i in zip(range(0, 7), [3, 5, 5, 3, 3, 5, 5]):
        item[sentence_i].insert(i, wordset['verb_argument'])

    for sentence_i, i in zip(range(7, 14), [3, 5, 6, 3, 4, 6, 5]):
        item[sentence_i].insert(i, wordset['verb_argument'])

    for sentence_i, i in zip(range(14, 21), [3, 5, 5, 3, 3, 5, 5]):
        item[sentence_i].insert(i, wordset['verb_argument'])

    for sentence_i, i in zip(range(21, 28), [3, 5, 6, 3, 4, 6, 5]):
        item[sentence_i].insert(i, wordset['verb_argument'])

    return item


def add_auxiliary_embedding(item):

    for sentence_i, i in zip(range(0, 7), [1, 3, 4, 1, 2, 4, 3]):
        item[sentence_i].insert(i, np1_pbe)
        item[sentence_i].insert(i+1, 'going to')

    for sentence_i, i in zip(range(7, 14), [2, 4, 5, 2, 3, 5, 4]):
        item[sentence_i].insert(i, 'going to')
        item[sentence_i].insert(i+1, 'be')

    for sentence_i, i in zip(range(14, 21), [1, 3, 4, 1, 2, 4, 3]):
        item[sentence_i].insert(i, np2_pbe)
        item[sentence_i].insert(i+1, 'going to')

    for sentence_i, i in zip(range(21, 28), [2, 4, 5, 2, 3, 5, 4]):
        item[sentence_i].insert(i, 'going to')
        item[sentence_i].insert(i+1, 'be')

    return item


def add_quotative_embedding(wordset, item):
    """Add embedding at top of sentence."""

    for sentence_i, i in zip(range(0, 7), [[0], [0, 3], [0, 3], [0, 1], [0, 1], [0, 3], [0, 3]]):
        i = random.sample(i, k=1)[0]

        item[sentence_i].insert(i, wordset['quotative_embedding'][0])

        that_placed = False
        if i is 0 and random.random() <= 0.50:
            item[sentence_i].insert(i+1, 'that')
            that_placed = True

    for sentence_i, i in zip(range(7, 14), [[0], [0, 3], [0, 3], [0, 1], [0, 1], [0, 3], [0, 3]]):
        i = random.sample(i, k=1)[0]

        item[sentence_i].insert(i, wordset['quotative_embedding'][0])
        
        that_placed = False
        if i is 0 and random.random() <= 0.50:
            item[sentence_i].insert(i+1, 'that')
            that_placed = True

    for sentence_i, i in zip(range(14, 21), [[0], [0, 3], [0, 3], [0, 1], [0, 1], [0, 3], [0, 3]]):
        i = random.sample(i, k=1)[0]

        item[sentence_i].insert(i, wordset['quotative_embedding'][0])

        that_placed = False
        if i is 0 and random.random() <= 0.50:
            item[sentence_i].insert(i+1, 'that')
            that_placed = True

    for sentence_i, i in zip(range(21, 28), [[0], [0, 3], [0, 3], [0, 1], [0, 1], [0, 3], [0, 3]]):
        i = random.sample(i, k=1)[0]

        item[sentence_i].insert(i, wordset['quotative_embedding'][0])
        
        that_placed = False
        if i is 0 and random.random() <= 0.50:
            item[sentence_i].insert(i+1, 'that')
            that_placed = True

    return item


def add_parenthetical(wordset, item):

    parenthetical_placement_indices = []

    parenthetical_affects_only_nps = False

    if wordset['parenthetical']['type'] == 'A':
        parenthetical_affects_only_nps = False
    elif wordset['parenthetical']['type'] == 'BC':
        parenthetical_affects_only_nps = True
    elif wordset['parenthetical']['type'] == 'ABC':
        if random.random() <= 0.33:
            parenthetical_affects_only_nps = False
        else:
            parenthetical_affects_only_nps = True

    if parenthetical_affects_only_nps:

        affecting_np1 = False
        if random.random() <= 0.50:
            affecting_np1 = True
        np1 = wordset['noun_phrases'][0]['words']
        np2 = wordset['noun_phrases'][1]['words']

        for sentence_i in range(0, 14):
            for i in range(len(item[sentence_i])):
                current_phrase = make_nom(item[sentence_i][i])

                if affecting_np1:
                    if current_phrase == np1:
                        parenthetical_placement_indices.append(i+1)
                        break
                else:
                    if current_phrase == np2:
                        parenthetical_placement_indices.append(i+1)
                        break
        # Halfway through item, switch the NP the parenthetical attaches to
        for sentence_i in range(14, 28):
            for i in range(len(item[sentence_i])):
                current_phrase = make_nom(item[sentence_i][i])

                if affecting_np1:
                    if current_phrase == np2:
                        parenthetical_placement_indices.append(i+1)
                        break
                else:
                    if current_phrase == np1:
                        parenthetical_placement_indices.append(i+1)
                        break
    else:
        parenthetical_placement_indices = [0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0,
                                           0, 0, 0, 0, 0, 0, 0
                                          ]
    
    for sentence_i, pth_i in zip(range(len(item)), parenthetical_placement_indices):
        item[sentence_i].insert(pth_i, ',')
        item[sentence_i].insert(pth_i+1, wordset['parenthetical']['words'])
        item[sentence_i].insert(pth_i+2, ',')
    
    return item

def write_extra_file(output_fp):

    with open(output_fp, 'w') as out:
        dyad = 0
        out.write('dyad,auxiliary_embedding,quotative_embedding,parenthetical,np1,verb,np2,verb_argument,prepositional_phrase\n')
        for wordset in wordsets:

            dyad += 1

            out.write('{},'.format(dyad))

            out.write('{},'.format(wordset['auxiliary_embedding']))

            if wordset['quotative_embedding']:
                out.write('"{}",'.format(wordset['quotative_embedding'][0]))
            else:
                out.write(',')

            if wordset['parenthetical']:
                out.write('"{}",'.format(wordset['parenthetical']['words']))
            else:
                out.write(',')

            out.write('"{}",'.format(wordset['noun_phrases'][0]['words']))
            out.write('"{}",'.format(wordset['verb']))
            out.write('"{}",'.format(wordset['noun_phrases'][1]['words']))

            if wordset['verb_argument']:
                out.write('"{}",'.format(wordset['verb_argument']))
            else:
                out.write(',')

            if wordset['prepositional_phrase']:
                out.write('"{}"\n'.format(wordset['prepositional_phrase']))
            else:
                out.write('\n')


def write_to_file(output_fp):

    sentence_labels = ['active,basic,a',
                       'active,subj-it-cleft,a',
                       'active,comp-it-cleft,a',
                       'active,subj-wh-cleft,a',
                       'active,comp-wh-cleft,a',
                       'active,subj-fronting,a',
                       'active,obj-fronting,a',
                       'passive,basic,a',
                       'passive,subj-it-cleft,a',
                       'passive,comp-it-cleft,a',
                       'passive,subj-wh-cleft,a',
                       'passive,comp-wh-cleft,a',
                       'passive,subj-fronting,a',
                       'passive,obj-fronting,a',
                       'active,basic,b',
                       'active,subj-it-cleft,b',
                       'active,comp-it-cleft,b',
                       'active,subj-wh-cleft,b',
                       'active,comp-wh-cleft,b',
                       'active,subj-fronting,b',
                       'active,obj-fronting,b',
                       'passive,basic,b',
                       'passive,subj-it-cleft,b',
                       'passive,comp-it-cleft,b',
                       'passive,subj-wh-cleft,b',
                       'passive,comp-wh-cleft,b',
                       'passive,subj-fronting,b',
                       'passive,obj-fronting,b',
                        'active,basic,c',
                        'active,subj-it-cleft,c',
                        'active,comp-it-cleft,c',
                        'active,subj-wh-cleft,c',
                        'active,comp-wh-cleft,c',
                        'active,subj-fronting,c',
                        'active,obj-fronting,c',
                        'passive,basic,c',
                        'passive,subj-it-cleft,c',
                        'passive,comp-it-cleft,c',
                        'passive,subj-wh-cleft,c',
                        'passive,comp-wh-cleft,c',
                        'passive,subj-fronting,c',
                        'passive,obj-fronting,c'
                      ]

    with open(output_fp, 'w') as out:

        id_num = 0
        dyad = 0

        out.write('id,dyad,voice,construction,set,sentence\n')

        for wordset in wordsets:

            dyad += 1

            num_label = -1

            for sentence in convert_sentences_to_strings(wordset['item']):
                id_num += 1
                num_label += 1

                    
                out.write('{},'.format(id_num))
                out.write('{},'.format(dyad))
                out.write('{},'.format(sentence_labels[num_label]))
                out.write('"{}"\n'.format(sentence))



                '''out.write('{},'.format(wordset['auxiliary_embedding']))

                if wordset['quotative_embedding']:
                    out.write('"{}",'.format(wordset['quotative_embedding']))
                else:
                    out.write(',')

                if wordset['parenthetical']:
                    out.write('"{}",'.format(wordset['parenthetical']['words']))
                else:
                    out.write(',')

                out.write('"{}",'.format(wordset['noun_phrases'][0]['words']))
                out.write('"{}",'.format(wordset['verb']))
                out.write('"{}",'.format(wordset['noun_phrases'][1]['words']))

                if wordset['verb_argument']:
                    out.write('"{}",'.format(wordset['verb_argument']))
                else:
                    out.write(',')

                if wordset['prepositional_phrase']:
                    out.write('"{}",'.format(wordset['prepositional_phrase']))
                else:
                    out.write(',')'''

                


if __name__ == "__main__":

    NP_FP = 'library/NPs.txt'
    VB_FP = 'library/verbs.txt'
    PP_FP = 'library/PPs.txt'
    PTH_FP = 'library/parentheticals.txt'
    EMD_FP = 'library/sentence_embeddings.txt'

    VERB_TENSE_DICT_FP = 'library/all_verb_tenses.json'

    PP_PROBABILITY = 0.10
    PTH_PROBABILITY = 0.10
    EMD_PROBABILITY = 0.20
    AUXEMD_PROBABILITY = 0.10

    #ITEMS_TO_GENERATE = int(sys.argv[2])
    
    try:
        OUTPUT_FILENAME = sys.argv[1]
    except IndexError:
        OUTPUT_FILENAME = 'sentence_generator'

    nps, verbs, pps, parenths, embeds, all_verbs = populate_libraries(NP_FP, VB_FP, PP_FP, PTH_FP, EMD_FP, VERB_TENSE_DICT_FP)

    np_list = []
    for np in nps:
        np_list.append(np['words'])
    print('==============================\n+ Total nouns: {}\n+ Total verbs: {}\n+ Total items: {}\n=============================='.format(len(np_list), len(verbs), int(len(np_list)/2) ))#, len(np_list), '\n+ Total items:', int(len(np_list)/2), '\n==============================')

    k = int(len(nps) * 0.10)
    indices = random.sample(range(0, len(nps)), k)
    nps_test = [nps[i] for i in indices] #1106
    nps_train = [nps[i] for i in range(len(nps)) if i not in indices] #9954

    k = int(len(verbs) * 0.10 + 1)
    indices = random.sample(range(len(verbs)), k)
    verbs_test = [verbs[i] for i in indices]
    verbs_train = [verbs[i] for i in range(len(verbs)) if i not in indices]

    k = int(len(pps) * 0.10 + 1)
    indices = random.sample(range(len(pps)), k)
    pps_test = [pps[i] for i in indices]
    pps_train = [pps[i] for i in range(len(pps)) if i not in indices]

    k = int(len(parenths) * 0.10 + 1)
    indices = random.sample(range(len(parenths)), k)
    parenths_test = [parenths[i] for i in indices]
    parenths_train = [parenths[i] for i in range(len(parenths)) if i not in indices]

    k = int(len(embeds) * 0.10 + 1)
    indices = random.sample(range(len(embeds)), k)
    embeds_test = [embeds[i] for i in indices]
    embeds_train = [embeds[i] for i in range(len(embeds)) if i not in indices]

    total_nouns_used = 0

    for output_type in ['train', 'test']:

        output_fp = OUTPUT_FILENAME + '.' + output_type + '.csv'
        if output_type == 'train':
            print('\n-----Generating training set:')
            nps = nps_train
            verbs = verbs_train
            '''pps = pps_train
            parenths = parenths_train
            embeds = embeds_train'''
            #num_of_items = ITEMS_TO_GENERATE
        else:
            print('\n\n-----Generating testing set:')
            nps = nps_test
            verbs = verbs_test
            '''pps = pps_test
            parenths = parenths_test
            embeds = embeds_test'''
            #num_of_items = int(ITEMS_TO_GENERATE / 10)
            #if not num_of_items:
            #    num_of_items = 1

        wordsets = []

        np_sets = []
        np_verb_sets = []

        ####
        # Get and sum all inverse probabilities
        probabilitiy_sum = 0
        initial_probabilities = []
        for np in nps:
            inverse = 1/len(np['words'])
            initial_probabilities.append(inverse)
            probabilitiy_sum += inverse
        
        # Find the true percentage value for each inverse probability
        final_probabilities = []
        for probability in initial_probabilities:
            final_probabilities.append(probability/probabilitiy_sum)
        
        # These percentages don't properly sum to 1.0; need to
        # arbitrarily add enough to one value in order to equal 1.0
        probabilitiy_sum = 0
        for probability in final_probabilities:
            probabilitiy_sum += probability
        remainder_to_add = 1 - probabilitiy_sum
        lucky_one = random.randint(0, len(final_probabilities)-1)

        # Change that value, then return the list of probabilities
        final_probabilities[lucky_one] = final_probabilities[lucky_one] + remainder_to_add
        ####
        if output_type == 'test':
            num = int((len(nps)/2))

            random.shuffle(nps)
            distributed = nps[:int((len(nps)/2))]
            remaining_nps = nps[int((len(nps)/2)):]

            total_nouns_used += len(distributed) + len(remaining_nps)

            random.shuffle(verbs)
            verb_i = 0
            for np, np2 in zip(distributed, remaining_nps):
                verb = verbs[verb_i]
                verb_i += 1
                if verb_i >= len(verbs):
                    random.shuffle(verbs)
                    verb_i = 0

                generate_wordsets(final_probabilities, np, np2, verb)
                print('Generating wordsets... ({}/{})'.format(len(wordsets), len(distributed)), end='\r')

        elif output_type == 'train':
            num = int((len(nps)/2))

            random.shuffle(nps)
            distributed = nps[:int((len(nps)/2))]
            remaining_nps = nps[int((len(nps)/2)):]

            total_nouns_used += len(distributed) + len(remaining_nps)

            random.shuffle(verbs)
            verb_i = 0
            for np, np2 in zip(distributed, remaining_nps):
                verb = verbs[verb_i]
                verb_i += 1
                if verb_i >= len(verbs):
                    random.shuffle(verbs)
                    verb_i = 0

                generate_wordsets(final_probabilities, np, np2, verb)
                print('Generating wordsets... ({}/{})'.format(len(wordsets), len(distributed)), end='\r')

        '''if num_of_items <= len(nps):
            distributed = random.sample(nps, k=num_of_items)
            remaining_nps = [x for x in nps if x not in distributed]
            remaining_nps = random.sample(remaining_nps, k=len(remaining_nps))
            print('Len of chosen np list:', len(distributed), '; Len of rest of NPs:', len(remaining_nps))
            print('+')

            for np, np2 in zip(distributed, remaining_nps):
                generate_wordsets(final_probabilities, np, np2)
                print('Generating wordsets... ({}/{})'.format(len(wordsets), num_of_items), end='\r')
        else:
            distributed = random.sample(nps, k=len(nps))
            for np in distributed:
                generate_wordsets(final_probabilities, np)
                print('Generating wordsets... ({}/{})'.format(len(wordsets), num_of_items), end='\r')

            for x in range(num_of_items - len(distributed)):
                generate_wordsets(final_probabilities)
                print('Generating wordsets... ({}/{})'.format(len(wordsets), num_of_items), end='\r')'''
        print()

        del np_sets
        del np_verb_sets
        del distributed

        for wordset in wordsets:
            print('Making sentences from wordsets... ({}/{})'.format(wordsets.index(wordset)+1, num), end='\r')

            np1, np2, np1_pbe, np2_pbe = assign_noun_phrase_agreement(wordset)
            verb, ppart_verb = assign_verb_conjugation(wordset, np1_pbe, np2_pbe)

            item = []
            on = False

            item.append([np1, verb, make_acc(np2)])
            item.append(['it was', np1, 'who', verb, make_acc(np2)])
            item.append(['it was', make_acc(np2), 'who', np1, verb])
            item.append(['who', verb, make_acc(np2), np1_pbe, make_acc(np1)])
            item.append(['who', np1, verb, np2_pbe, make_acc(np2)])
            item.append([make_acc(np2), np2_pbe, 'who', np1, verb])
            item.append([np1, np1_pbe, 'who', verb, make_acc(np2)])

            item.append([np2, np2_pbe, ppart_verb, 'by', make_acc(np1)])
            item.append(['it was', np2, 'who', np2_pbe, ppart_verb, 'by', make_acc(np1)])
            item.append(['it was', make_acc(np1), 'who', np2, np2_pbe, ppart_verb, 'by'])
            item.append(['who', 'was', ppart_verb, 'by', make_acc(np1), np2_pbe, make_acc(np2)])
            item.append(['who', np2, np2_pbe, ppart_verb, 'by', np1_pbe, make_acc(np1)])
            item.append([np1, np1_pbe, 'who', np2, np2_pbe, ppart_verb, 'by'])
            item.append([np2, np2_pbe, 'who', np2_pbe, ppart_verb, 'by', make_acc(np1)])

            item.append([np2, verb, make_acc(np1)])
            item.append(['it was', np2, 'who', verb, make_acc(np1)])
            item.append(['it was', make_acc(np1), 'who', np2, verb])
            item.append(['who', verb, make_acc(np1), np2_pbe, make_acc(np2)])
            item.append(['who', np2, verb, np1_pbe, make_acc(np1)])
            item.append([make_acc(np1), np1_pbe, 'who', np2, verb])
            item.append([np2, np2_pbe, 'who', verb, make_acc(np1)])

            item.append([np1, np1_pbe, ppart_verb, 'by', make_acc(np2)])
            item.append(['it was', np1, 'who', np1_pbe, ppart_verb, 'by', make_acc(np2)])
            item.append(['it was', make_acc(np2), 'who', np1, np1_pbe, ppart_verb, 'by'])
            item.append(['who', 'was', ppart_verb, 'by', make_acc(np2), np1_pbe, make_acc(np1)])
            item.append(['who', np1, np1_pbe, ppart_verb, 'by', np2_pbe, make_acc(np2)])
            item.append([np2, np2_pbe, 'who', np1, np1_pbe, ppart_verb, 'by'])
            item.append([np1, np1_pbe, 'who', np1_pbe, ppart_verb, 'by', make_acc(np2)])

            if wordset['quotative_embedding']:
                if wordset['quotative_embedding'][1] != '-1':

                    # Set embedded NP
                    emb_np = wordset['quotative_embedding'][0].rsplit(' ', 1)[0]
                    emb_vb = wordset['quotative_embedding'][0].rsplit(' ', 1)[1]

                    # Choose version of Set C to generate
                    if random.random() > 0.5:
                        temp = np2

                        np1 = np1
                        np2 = emb_np
                        emb_np = temp

                        np1_pbe = np1_pbe

                        if wordset['noun_phrases'][1]['plurality'] == '0':
                            if all_verbs.get(emb_vb):
                                emb_vb = all_verbs[emb_vb][1]

                    else:
                        temp = np1

                        np1 = np2
                        np2 = emb_np
                        emb_np = temp

                        np1_pbe = np2_pbe

                        if wordset['noun_phrases'][0]['plurality'] == '0':
                            if all_verbs.get(emb_vb):
                                emb_vb = all_verbs[emb_vb][1]

                    if wordset['quotative_embedding'][1] == '0':
                        np2_pbe = 'was'
                    elif wordset['quotative_embedding'][1] == '1':
                        np2_pbe = 'were'                        

                    new_embedding = emb_np + ' ' + emb_vb

                    item.append([np1, verb, make_acc(np2)])
                    item.append(['it was', np1, 'who', verb, make_acc(np2)])
                    item.append(['it was', make_acc(np2), 'who', np1, verb])
                    item.append(['who', verb, make_acc(np2), np1_pbe, make_acc(np1)])
                    item.append(['who', np1, verb, np2_pbe, make_acc(np2)])
                    item.append([make_acc(np2), np2_pbe, 'who', np1, verb])
                    item.append([np1, np1_pbe, 'who', verb, make_acc(np2)])

                    item.append([np2, np2_pbe, ppart_verb, 'by', make_acc(np1)])
                    item.append(['it was', np2, 'who', np2_pbe, ppart_verb, 'by', make_acc(np1)])
                    item.append(['it was', make_acc(np1), 'who', np2, np2_pbe, ppart_verb, 'by'])
                    item.append(['who', 'was', ppart_verb, 'by', make_acc(np1), np2_pbe, make_acc(np2)])
                    item.append(['who', np2, np2_pbe, ppart_verb, 'by', np1_pbe, make_acc(np1)])
                    item.append([np1, np1_pbe, 'who', np2, np2_pbe, ppart_verb, 'by'])
                    item.append([np2, np2_pbe, 'who', np2_pbe, ppart_verb, 'by', make_acc(np1)])

                    on = True

            if wordset['prepositional_phrase']:
                for sentence in item:
                    sentence.append(wordset['prepositional_phrase'])

            if wordset['verb_argument']:
                item = add_verb_argument(wordset, item)

                if on:
                    for sentence_i, i in zip(range(28, 35), [3, 5, 5, 3, 3, 5, 5]):
                        item[sentence_i].insert(i, wordset['verb_argument'])

                    for sentence_i, i in zip(range(35, 42), [3, 5, 6, 3, 4, 6, 5]):
                        item[sentence_i].insert(i, wordset['verb_argument'])

            if wordset['auxiliary_embedding']: 
                item = add_auxiliary_embedding(item)

                if on:
                    for sentence_i, i in zip(range(28, 35), [1, 3, 4, 1, 2, 4, 3]):
                        item[sentence_i].insert(i, np1_pbe)
                        item[sentence_i].insert(i+1, 'going to')

                    for sentence_i, i in zip(range(35, 42), [2, 4, 5, 2, 3, 5, 4]):
                        item[sentence_i].insert(i, 'going to')
                        item[sentence_i].insert(i+1, 'be')

            if wordset['quotative_embedding']:
                item = add_quotative_embedding(wordset, item)

                if on:
                    for sentence_i, i in zip(range(28, 35), [[0], [0, 3], [0, 3], [0, 1], [0, 1], [0, 3], [0, 3]]):
                        i = random.sample(i, k=1)[0]

                        item[sentence_i].insert(i, new_embedding)

                        that_placed = False
                        if i is 0 and random.random() <= 0.50:
                            item[sentence_i].insert(i+1, 'that')
                            that_placed = True

                    for sentence_i, i in zip(range(35, 42), [[0], [0, 3], [0, 3], [0, 1], [0, 1], [0, 3], [0, 3]]):
                        i = random.sample(i, k=1)[0]

                        item[sentence_i].insert(i, new_embedding)
                        
                        that_placed = False
                        if i is 0 and random.random() <= 0.50:
                            item[sentence_i].insert(i+1, 'that')
                            that_placed = True

            if wordset['parenthetical']:
                item = add_parenthetical(wordset, item)

                if on:
                    parenthetical_placement_indices = []

                    parenthetical_affects_only_nps = False

                    if wordset['parenthetical']['type'] == 'A':
                        parenthetical_affects_only_nps = False
                    elif wordset['parenthetical']['type'] == 'BC':
                        parenthetical_affects_only_nps = True
                    elif wordset['parenthetical']['type'] == 'ABC':
                        if random.random() <= 0.33:
                            parenthetical_affects_only_nps = False
                        else:
                            parenthetical_affects_only_nps = True

                    if parenthetical_affects_only_nps:

                        affecting_np1 = False
                        if random.random() <= 0.50:
                            affecting_np1 = True

                        for sentence_i in range(28, 42):
                            for i in range(len(item[sentence_i])):
                                current_phrase = make_nom(item[sentence_i][i])

                                if affecting_np1:
                                    if current_phrase == np1:
                                        parenthetical_placement_indices.append(i+1)
                                        break
                                else:
                                    if current_phrase == np2:
                                        parenthetical_placement_indices.append(i+1)
                                        break

                    else:
                        parenthetical_placement_indices = [0, 0, 0, 0, 0, 0, 0,
                                                           0, 0, 0, 0, 0, 0, 0,
                                                          ]
                    
                    for sentence_i, pth_i in zip(range(28, 42), parenthetical_placement_indices):
                        item[sentence_i].insert(pth_i, ',')
                        item[sentence_i].insert(pth_i+1, wordset['parenthetical']['words'])
                        item[sentence_i].insert(pth_i+2, ',')


            wordset['item'] = item

        write_to_file(output_fp)

        try:
            if sys.argv[2] == 'parts':
                write_extra_file(OUTPUT_FILENAME + '.' + output_type + '.parts.csv')
        except:
            pass

    print()
    print()
    
