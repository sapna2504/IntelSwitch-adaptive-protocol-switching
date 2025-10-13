import datetime
import os
import time
import logging
import numpy as np
import multiprocessing as mp
import sys
import threading
import json
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
import gc
import env
import math

sys.path.insert(0, "../util")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import tensorflow as tf
import a3c_IntelDASH_copy

# Global parameters

S_INFO = 3  # bit_rate, bitrate_variation, rebuffering and networkthroughput
S_LEN = 8  # take how many frames in the past
A_DIM = 2
ACTOR_LR_RATE = 0.0003
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 6 
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 1000 # in epoch
BUFFER_NORM_FACTOR = 30.0
M_IN_K = 1000.0
REBUF_PENALTY = 700  # 1 sec rebuffering -> 4.3 Mbps ->In our case it is 700
SMOOTH_PENALTY = 1
NORM_VAL = 700
DEFAULT_PROTOCOL = 'http3'  # default video protocol without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = '/mnt/hdd/rl_results_inteldash'
LOG_FILE = '/mnt/hdd/rl_results_inteldash/log'
TEST_LOG_FOLDER = '/mnt/hdd/rl_results_inteldash/test_results/'
NN_MODEL = '/mnt/hdd/rl_results_inteldash/nn_model_ep_4000.ckpt'
# NN_MODEL = None
#PORT = [8222, 8333, 8444, 8555, 8666, 8777, 8888]
#PROTO = [0,1,2,2,2,2]
PORT = [8666, 8333, 8777]
NETWORK = ['4g', '5g', 'mmwave-mid-band-drive', 'wifi', 'ethernet', 'wifi-poor']
COOKED_DATA_FOLDER = './hotnets_data_new/'



PROTOCOL_MAP = {
    'http2': 0,
    'http3': 1
}

def mlog(fnc="none", msg=""):
    print(msg)

def to_bool(val):
    return str(val).strip().lower() == 'true'

def central_agent(net_params_queue, exp_queues):

    #tf.compat.v1.reset_default_graph() 
    assert len(net_params_queue) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    
    # initial_entropy_weight = 0.005
    # final_entropy_weight = 0.001
    # entropy_decay = 0.998

    # entropy_weight = initial_entropy_weight

    with sess, open(LOG_FILE + '_test', 'w') as test_log_file:

        
        actor = a3c_IntelDASH_copy.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c_IntelDASH_copy.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c_IntelDASH_copy.build_summaries()

        sess.run(tf.compat.v1.global_variables_initializer())
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        TRAIN_SUMMARY_DIR = os.path.join(SUMMARY_DIR, curr_time, 'train')
        TEST_SUMMARY_DIR = os.path.join(SUMMARY_DIR, curr_time, 'test')
        writer = tf.compat.v1.summary.FileWriter(TRAIN_SUMMARY_DIR, sess.graph)  # training monitor
        test_writer = tf.compat.v1.summary.FileWriter(TEST_SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.compat.v1.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        
        # Initialize global video count
        epoch = 4001
        test_avg_reward, test_avg_entropy, test_avg_td_loss = 0, 0.5, 0

        # if epoch < 1000:
        #     entropy_weight = 0.1  # Keep high exploration
            
        # assemble experiences from agents, compute the gradients
        while True:
            # record average reward and td loss change
            # in the experiences from the agents
            
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in range(NUM_AGENTS):
                try:
                    net_params_queue[i].put([actor_net_params, critic_net_params])
                except queue.Full:
                    print(f"[WARNING] Queue to agent {i} is full. Skipping update.")   
            
            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []
            s_batch = None
            a_batch = None
            r_batch = None
            terminal = None
            entropy = None

            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0
            
            for i in range(NUM_AGENTS):
                try:
                    s_batch, a_batch, r_batch, terminal, entropy = exp_queues[i].get(timeout=1.0)

                    if len(r_batch) == 0:
                        print(f"[WARNING] Skipping empty batch from agent {i}")
                        continue

                    actor_gradient, critic_gradient, td_batch, old_log_probs = \
                    a3c_IntelDASH_copy.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)
                    # print("44444444444444444")
                    actor_gradient_batch.append(actor_gradient)
                    critic_gradient_batch.append(critic_gradient)

                    total_reward += np.sum(r_batch)
                    total_td_loss += np.sum(td_batch)
                    total_batch_len += len(r_batch)
                    total_agents += 1.0
                    total_entropy += np.sum(entropy)
                    print("the terminal value is ", terminal, total_reward, total_td_loss, total_entropy, total_batch_len, total_agents)                               

                except queue.Empty:
                    continue
            
            #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa", len(actor_gradient_batch))
            # compute aggregated gradient
           
        
            for i in range(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            gc.collect()
            
            #print("the terminal value is before if ", total_reward, total_td_loss, total_entropy, total_batch_len, total_agents) 
            if total_agents > 0 and total_batch_len >0:
                avg_reward = total_reward/total_agents
                avg_td_loss = total_td_loss/total_batch_len
                avg_entropy = total_entropy/total_batch_len
                epoch += 1

                # entropy_weight = max(final_entropy_weight, entropy_weight * entropy_decay)
                # a3c_IntelDASH_copy.ENTROPY_WEIGHT = entropy_weight

                print(f"[Epoch {epoch}] TD_loss={avg_td_loss}, Reward={avg_reward}, Entropy={avg_entropy}")

            
            
                logging.info('Epoch: ' + str(epoch) +
                            ' TD_loss: ' + str(avg_td_loss) +
                            ' Avg_reward: ' + str(avg_reward) +
                            ' Avg_entropy: ' + str(avg_entropy))
                
                #Training summary
                if summary_ops is not None:
                    summary_str = sess.run(summary_ops, feed_dict={
                        summary_vars[0]: avg_td_loss,
                        summary_vars[1]: avg_reward,
                        summary_vars[2]: avg_entropy
                    })
                writer.add_summary(summary_str, epoch)
                writer.flush()
                # Testing summary
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: test_avg_td_loss,
                    summary_vars[1]: test_avg_reward,
                    summary_vars[2]: test_avg_entropy
                })

                test_writer.add_summary(summary_str, epoch)
                test_writer.flush()
                if epoch % MODEL_SAVE_INTERVAL == 0 and epoch != 0:
                    # Save the neural net parameters to disk.
                    save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
                    logging.info("Model saved in file: " + save_path)
                    gc.collect()
                    #test_avg_reward, test_avg_entropy = testing(epoch, save_path + ".ckpt", test_log_file)
            else:
                pass


def agent(agent_id, data_tcp, data_quic, net_params_queue, exp_queue, log_file_path=LOG_FILE):

    net_env = env.Environment(data_tcp=data_tcp,
                              data_quic=data_quic,
                              agent_id=agent_id)
    
    tf.compat.v1.reset_default_graph()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    with sess, open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
        
        actor = a3c_IntelDASH_copy.ActorNetwork(sess,
                                    state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                    learning_rate=ACTOR_LR_RATE)
        critic = a3c_IntelDASH_copy.CriticNetwork(sess,
                                    state_dim=[S_INFO, S_LEN],
                                    learning_rate=CRITIC_LR_RATE)
        
        sess.run(tf.compat.v1.global_variables_initializer()) 
        try:
            actor_net_params, critic_net_params = net_params_queue.get(timeout=0.01)
            actor.set_network_params(actor_net_params)
            critic.set_network_params(critic_net_params)
        except queue.Empty:
            print("[INFO] net_params_queue is empty. Skipping param update.")

        # last_protocol = DEFAULT_PROTOCOL
        #protocol here will come from DASH and not by indexing into some array
        #protocol = DEFAULT_PROTOCOL 
        protocol = 1  # 0 = TCP, 1 = QUIC
        last_protocol = protocol

        action_vec = np.zeros(A_DIM)
        #protocol_index = PROTOCOL_MAP.get(protocol, 0)  # Default to 0 if unknown 
        action_vec[protocol] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []
        throughput_history = []
        # field_names = ['time', 'segment_index', 'currentBitrate', 'lastBitrate', 'RebufferTime', 'throughput', 'bitrateVariation', 'buffer', 'droppedFrames', 'cpu_pressure', 'mem_pressure', 'latency', 'networkThroughput', 'playbackEnded', 'protocol']
        field_names = ['time', 'segment_index', 'currentBitrate', 'lastBitrate', 'RebufferTime', 'throughput', 'bitrateVariation', 'buffer', 'droppedFrames', 'cpu_pressure', 'mem_pressure', 'latency', 'playbackEnded', 'protocol']
        post_data = {}
        send_data = 1

        last_total_rebuf = 0 
        input_dict = {'sess': sess,
                      'actor': actor, 'critic': critic, 
                      'log_file': log_file,
                      'last_protocol': last_protocol,
                      'last_total_rebuf': last_total_rebuf,
                      's_batch': s_batch, 'a_batch': a_batch, 'r_batch': r_batch,
                      'log_file_path': log_file_path, 'cpu_pressure': 0, 'mem_pressure': 0 , 'entropy_record': entropy_record, 'send_data': send_data, 'throughput_history': throughput_history}

        time_stamp = 0
        rolling_std = 0
        norm_std = 0
        

        while True:  # experience video streaming forever

            for i in range(150):
                try:
                    row = net_env.get_next_match_cyclic(i, input_dict['send_data'])
                    # print(f"Row for {i}: {row}, type: {type(row)}")
                    if row is None:
                        print(f"No match for {i}")
                        continue
                    # handle case: multiple matches returned
                    # if isinstance(row, tuple):
                    #     print(f"[INFO] Multiple matches for {i}, taking last one")
                    #     print(f"Row content: {row}")
                    #     row = row[-1]
                    if len(field_names) != len(row):
                        print(f"[ERROR] Mismatch: len(field_names)={len(field_names)}, len(row)={len(row)}")
                        # print(f"Row content: {row}")
                        continue
                    post_data = dict(zip(field_names, row))
                    print(post_data)
                except ValueError as e:
                    print(f"Warning: {e}")
                       
                
                # rebuffer_time = float(post_data['RebufferTime'])
                # # rebuffer_penalty = REBUF_PENALTY * (rebuffer_time /10)
                # rebuffer_penalty = REBUF_PENALTY * (rebuffer_time)
                
                # reward = \
                #     ((post_data['lastBitrate']/ M_IN_K) \
                #     - (rebuffer_penalty) \
                #     - (SMOOTH_PENALTY * (np.abs(post_data['bitrateVariation'])/ M_IN_K )))

                rebuffer_time = float(post_data['RebufferTime'])
                # rebuffer_penalty = REBUF_PENALTY * (rebuffer_time /3)
                rebuffer_penalty = REBUF_PENALTY * (rebuffer_time)   
                print("he bitrate is ", post_data['currentBitrate'], post_data['bitrateVariation'], rebuffer_time )
                bitrate = math.log2(post_data['currentBitrate'])
                
                bitrate_variation = math.log2(1+abs(post_data['bitrateVariation']))
                rebuffering = math.log2(1+rebuffer_penalty)
                # else:
                #     bitrate_variation = post_data_tcp['bitrateVariation']
                #     rebuffering = rebuffer_penalty
                reward = bitrate - rebuffering 
                
                print("the reward is: ", reward)
                    

                rebuffer_time = 0  
                input_dict['r_batch'].append(reward)
                

                last_protocol = input_dict['last_protocol']
                end_of_video = False

                # retrieve previous state
                if len(input_dict['s_batch']) == 0:
                    state = np.zeros((S_INFO, S_LEN))
                else:
                    state = np.array(input_dict['s_batch'][-1], copy=True)

                # print("yha tak aeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")

                if post_data['cpu_pressure'] >10 and post_data['cpu_pressure'] < 30:
                    post_data['cpu_pressure'] = 50
                elif post_data['cpu_pressure'] >= 30:
                    post_data['cpu_pressure'] = 80
                else:
                    post_data['cpu_pressure'] = 5

                input_dict['throughput_history'].append(post_data['throughput']/M_IN_K)  # in Mbps
                if len(input_dict['throughput_history']) > 30:  # N = window size, e.g., 10
                    input_dict['throughput_history'].pop(0)
                
                if len(input_dict['throughput_history']) >= 30:
                    rolling_std = np.std(input_dict['throughput_history'])
                    norm_std = rolling_std / 246  # or whatever global max you’ve profiled
                else:
                    norm_std = 0  # or np.nan or previous value

                # dequeue history record
                state = np.roll(state, -1, axis=1)
            
                state[0, -1] = (post_data['currentBitrate']/M_IN_K) / NORM_VAL  # Normalize bitrate
                state[1, -1] = (post_data['RebufferTime']) / 3 # Normalize rebuffer time assuming 10 sec is the maximum rebuffering
                # state[2, -1] = (post_data['cpu_pressure']) / 30
                state[2, -1] = (post_data['throughput']/M_IN_K) /1000  # Normalize throughput assuming maximum throughput possible is 1000Mbps and obtained throughput in Kbps
                # state[3, -1] = norm_std
                
                
                # compute action probability vector
                
                
                # compute action probability vector
                action_prob = input_dict['actor'].predict(np.reshape(state, (1, S_INFO, S_LEN)))
                print("the action prob is: ", action_prob, agent_id)
                action_cumsum = np.cumsum(action_prob)
                protocol = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                input_dict['send_data'] = protocol
            
                # action_prob = input_dict['actor'].predict(np.reshape(state, (1, S_INFO, S_LEN)))
                # print("the action prob is: ", action_prob)
                # action_cumsum = np.cumsum(action_prob)
                # protocol = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                # #protocol = np.random.choice(range(A_DIM), p=action_prob[0])
                # input_dict['send_data'] = protocol


                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                    
                # because there is an intrinsic discrepancy in passing single state and batch states  
                # print("action-in-compute-entropy:",action_prob[0] )                         
                entropy_record.append(a3c_IntelDASH_copy.compute_entropy(action_prob[0]))

                end_of_video = post_data.get('playbackEnded', False)
                end_of_video = to_bool(end_of_video)
                # print('post_data = ', end_of_video)
                

                if len(input_dict['r_batch']) >= TRAIN_SEQ_LEN or end_of_video:
                # if end_of_video:
                    # print("3333333")
                    # print("yha aya sample collect ke baaaaaaaaaaaddd okkkokkokk")

                    exp_queue.put([input_dict['s_batch'][1:], 
                                input_dict['a_batch'][1:], 
                                input_dict['r_batch'][1:], 
                                end_of_video, 
                                input_dict['entropy_record'][1:]], timeout=0.01)

                    print("[INFO] Successfully pushed batch to exp_queue.",agent_id)

                    try:
                        actor_net_params, critic_net_params = net_params_queue.get(timeout=0.01)
                        actor.set_network_params(actor_net_params)
                        critic.set_network_params(critic_net_params)                              
                    except queue.Empty:
                        print("[INFO] net_params_queue is empty. Skipping param update.") 
        

                    del input_dict['s_batch'][:]
                    del input_dict['a_batch'][:]
                    del input_dict['r_batch'][:]
                    del input_dict['entropy_record'][:]
                    gc.collect()
                    print("[INFO] Training on 50 samples done. Continuing video playback.")


                if end_of_video:
                    # print('end-of-video mai aya')

                    protocol = 1
                    input_dict['last_protocol'] = protocol
                
                    action_vec = np.zeros(A_DIM)
                    action_vec[protocol] = 1

                    input_dict['s_batch'].append(np.zeros((S_INFO, S_LEN)))
                    input_dict['a_batch'].append(action_vec)
                    
                            

                else:
                    input_dict['s_batch'].append(state)
                    print("the protocol str value is ", protocol)
                    action_vec = np.zeros(A_DIM)
                    action_vec[protocol] = 1
                    input_dict['a_batch'].append(action_vec)               
            
            send_data = str(send_data)

            if len(send_data) > 0:
                mlog(fnc="do_POST()", msg="Response to POST req: {}".format(input_dict['send_data']))


    print("the agent {agent_id} has started")



def main():

    np.random.seed(RANDOM_SEED)

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []



    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(100000))
        exp_queues.append(mp.Queue(100000))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []

    for i in range(NUM_AGENTS):
        file_path_tcp = os.path.join(COOKED_DATA_FOLDER, NETWORK[i] + '-tcp' + '.csv')
        # print("theeeeeeeeeeeeeeeeeee", file_path_tcp)
        file_path_quic = os.path.join(COOKED_DATA_FOLDER, NETWORK[i] + '-quic' +  '.csv')
        try:
            data_tcp = np.loadtxt(file_path_tcp, dtype=str,  delimiter=',') 
            data_quic = np.loadtxt(file_path_quic, dtype=str,  delimiter=',')
            
        except Exception as e:
            print(f"Error loading files: {e}")
        
        agents.append(mp.Process(target=agent,
                                args=(i, data_tcp, data_quic,
                                    net_params_queues[i],
                                    exp_queues[i])))
        # print("aaaaaaaaaaaaaaaaaaaaaaa agent", i)


    for i in range(NUM_AGENTS):
        print('starting', i)
        agents[i].start()

        print("[INFO] All processes started.")


    # wait unit training is done
    #coordinator.join()

    try:
        while True:
            time.sleep(10)  # monitor every 10 seconds

            if not coordinator.is_alive():
                print("[ERROR] Central agent process has died!")
                break

            for i, ag in enumerate(agents):
                if not ag.is_alive():
                    print(f"[ERROR] Agent process {i} has died!")
                    break

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt caught. Exiting...")

    finally:
        print("[INFO] Terminating all processes...")

        coordinator.terminate()
        for ag in agents:
            ag.terminate()

        coordinator.join()
        for ag in agents:
            ag.join()

        print("[INFO] All processes terminated.")


if __name__ == '__main__':
    mp.set_start_method("fork")
    main()
