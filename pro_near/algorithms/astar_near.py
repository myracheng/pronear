import copy
import time
import random
import pickle
from .core import ProgramLearningAlgorithm, ProgramNodeFrontier
from program_graph import ProgramGraph
import numpy as np
import os
# import matplotlib.pyplot as plt
from utils.logging import log_and_print, print_program, print_program_dict
from utils.training import execute_and_train, execute_and_train_with_full,change_key,execute_and_train_og
from utils.logging import log_and_print,print_program
from pprint import pprint
from cpu_unpickle import traverse, CPU_Unpickler

### random seed
os.environ['PYTHONHASHSEED']='0'
np.random.seed(0)


class ASTAR_NEAR(ProgramLearningAlgorithm):

    def __init__(self, frontier_capacity=float('inf')):
        self.frontier_capacity = frontier_capacity

    def run_init(self, timestamp, base_program_name, hole_node_ind, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        log_and_print("Training root program ...")
        current = copy.deepcopy(graph.root_node)
        initial_score, losses, m = execute_and_train_with_full(base_program_name, hole_node_ind, current.program, validset, trainset, train_config, 
            graph.output_type, graph.output_size, neural=True, device=device)
        
        log_and_print("Initial training complete. Score from program is {:.4f} \n".format(1 - initial_score))

        if device == 'cpu':
            base_program = CPU_Unpickler(open("%s.p" % base_program_name, "rb")).load()
        else:
            base_program = pickle.load(open("%s.p" % base_program_name, "rb"))

        curr_level = 0
        l = []
        traverse(base_program.submodules,l)
        # pprint(l)
        curr_program = base_program.submodules

        change_key(base_program.submodules, [], hole_node_ind, current.program.submodules["program"])

        new_prog = base_program
        return 1 - initial_score, new_prog, losses

    def run_train_longer(self, timestamp, base_program_name, hole_node_ind, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        log_and_print("Training root program ...")
        current = copy.deepcopy(graph.root_node)
        initial_score, program = execute_and_train_og(base_program_name, validset, trainset, train_config, 
            graph.output_type, graph.output_size, neural=True, device=device)
        log_and_print("Re-training complete. Score from program is {:.4f} \n".format(1 - initial_score))
        return [{
                            "program" : program,
                            "score" : 1- initial_score,
                        }]

    def run(self, timestamp, base_program_name, hole_node_ind, graph, trainset, validset, train_config, device, verbose=False):
        assert isinstance(graph, ProgramGraph)

        log_and_print("Training root program ...")
        current = copy.deepcopy(graph.root_node)
        initial_score, l, m = execute_and_train_with_full(base_program_name, hole_node_ind, current.program, validset, trainset, train_config, 
            graph.output_type, graph.output_size, neural=True, device=device)
        # print("initial losses:")
        # print(l)
        # print("initial f1:")
        # print(m)
        log_and_print("Initial training complete. Score from program is {:.4f} \n".format(1 - initial_score))
        
        order = 0
        frontier = ProgramNodeFrontier(capacity=self.frontier_capacity) #mcheng priority queue
        frontier.add((float('inf'), order, current))
        num_children_trained = 0
        start_time = time.time()

        best_program = None
        best_total_cost = float('inf')
        best_programs_list = []

        # if not os.path.exists(timestamp):
        #     os.makedirs(timestamp)

        while len(frontier) != 0:
            current_f_score, _, current = frontier.pop(0)
            log_and_print("CURRENT program has fscore {:.4f}: {}".format(
                current_f_score, print_program(current.program, ignore_constants=(not verbose))))
            log_and_print("Current depth of program is {}".format(current.depth))
            log_and_print("Creating children for current node/program")
            children_nodes = graph.get_all_children(current)
            # print(children_nodes)
            # prune if more than self.max_num_children
            truncated_children = []
            symbolic_children = []
            if len(children_nodes) > graph.max_num_children:
                #keep all neural ones
                for c in children_nodes:
                    is_neural = not graph.is_fully_symbolic(c.program)
                    if is_neural:
                        truncated_children.append(c)
                    else: 
                        symbolic_children.append(c)
                n = len(truncated_children)
                if n < graph.max_num_children:
                    #get weights for each child
                    # if print_program(current.program).split('(')[-1].split(')')[0] == 'AtomToAtomModule':
                    #     weights = []
                    #     # print(weights_dict)
                    #     for c in symbolic_children:
                    #         diff_node = print_program(c.program).split('(')[-2]
                    #         # print(print_program(c.program))
                    #         # print(diff_node)
                    #         # try:
                    #         # weights.append(weights_dict[diff_node])
                    #         # except IndexError:
                    #             # print('tutu')
                    #             # weights.append(0)

                    # else: #weight equally
                    #     weights = [1] * len(symbolic_children)

                    # #make into probabilities
                    # sum_h = sum(weights)
                    # probs = [i/sum_h for i in weights]
                    # print(probs)

                    picked_children = np.random.choice(symbolic_children, graph.max_num_children - n)
                    #p=probs
                    truncated_children.extend(picked_children)
                    # truncated_children.extend(random.sample(symbolic_children, k=graph.max_num_children-n))  # sample without replacement
                else:
                    print(truncated_children)
                    truncated_children = random.sample(truncated_children, k=graph.max_num_children)
                children_nodes = truncated_children
            print(children_nodes)
            #todo if theres more neural children than alloewd.... 
            log_and_print("{} total children to train for current node".format(len(children_nodes)))

            for child_node in children_nodes:
                child_start_time = time.time()
                log_and_print("Training child program: {}".format(print_program(child_node.program, ignore_constants=(not verbose))))
                is_neural = not graph.is_fully_symbolic(child_node.program) #mcheng is not complete
                child_node.score, l, m = execute_and_train_with_full(base_program_name, hole_node_ind, child_node.program, validset, trainset, train_config, 
                    graph.output_type, graph.output_size, neural=is_neural, device=device)
                ## print losses and 1-f1 score for training
                # plt.close()
                # plt.figure()
                # plt.plot(l[2:])
                # plt.title("losses %s" % print_program(child_node.program, ignore_constants=(not verbose)))
                # plt.savefig("%s/losses_%s.png" % (timestamp,print_program(child_node.program, ignore_constants=(not verbose))))

                # plt.close()
                # plt.figure()
                # plt.plot(m[2:])
                # plt.title("f1 %s" %print_program(child_node.program, ignore_constants=(not verbose)))
                # plt.savefig("%s/f1_%s.png" % (timestamp,print_program(child_node.program, ignore_constants=(not verbose))))

                log_and_print("Time to train child {:.3f}".format(time.time() - child_start_time))
                num_children_trained += 1
                log_and_print("{} total children trained".format(num_children_trained))
                child_node.parent = current
                child_node.children = []
                order -= 1
                child_node.order = order  # insert order of exploration as tiebreaker for equivalent f-scores
                current.children.append(child_node)

                # computing path costs (f_scores)
                child_f_score = child_node.cost + child_node.score # cost + heuristic
                log_and_print("DEBUG: f-score {}".format(child_f_score))

                if not is_neural and child_f_score < best_total_cost:
                    best_program = copy.deepcopy(child_node.program)
                    best_total_cost = child_f_score
                    best_programs_list.append({
                            "program" : best_program,
                            "struct_cost" : child_node.cost, 
                            "score" : child_node.score,
                            "path_cost" : child_f_score,
                            "time" : time.time()-start_time
                        })
                    log_and_print("New BEST program found:")
                    print_program_dict(best_programs_list[-1])

                if is_neural: 
                    assert child_node.depth < graph.max_depth
                    child_tuple = (child_f_score, order, child_node)
                    frontier.add(child_tuple)

            # clean up frontier
            frontier.sort(tup_idx=0)
            while len(frontier) > 0 and frontier.peek(-1)[0] > best_total_cost:
                frontier.pop(-1)
            log_and_print("Frontier length is: {}".format(len(frontier)))
            log_and_print("Total time elapsed is {:.3f}".format(time.time()-start_time))

        if best_program is None:
            log_and_print("ERROR: no program found")

        return best_programs_list

#todo look at train fn, todo understand why some of the nodes are neural
#starts with a neural fn.