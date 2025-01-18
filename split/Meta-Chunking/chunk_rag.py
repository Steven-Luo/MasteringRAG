from perplexity_chunking import Chunking
from typing import List, Dict
import re
import math 
from nltk.tokenize import sent_tokenize
import jieba 
import torch

def split_text_by_punctuation(text,language): 
    if language=='zh': 
        sentences = jieba.cut(text, cut_all=False)  
        sentences_list = list(sentences)  
        sentences = []  
        temp_sentence = ""  
        for word in sentences_list:  
            if word in ["。", "！", "？","；"]:  
                sentences.append(temp_sentence.strip()+word)  
                temp_sentence = ""  
            else:  
                temp_sentence += word  
        if temp_sentence:   
            sentences.append(temp_sentence.strip())  
        
        return sentences
    else:
        full_segments = sent_tokenize(text)
        ret = []
        for item in full_segments:
            item_l = item.strip().split(' ')
            if len(item_l) > 512:
                if len(item_l) > 1024:
                    item = ' '.join(item_l[:256]) + "..."
                else:
                    item = ' '.join(item_l[:512]) + "..."
            ret.append(item)
        return ret


def find_minima(values,threshold):  
    minima_indices = []  
    for i in range(1, len(values) - 1):  
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            if (values[i - 1]-values[i]>=threshold) or (values[i + 1]-values[i]>=threshold):
                minima_indices.append(i)  
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            if values[i - 1]-values[i]>=threshold:
                minima_indices.append(i) 
    return minima_indices

def find_minima_dynamic(values,threshold,threshold_zlist):  
    minima_indices = []  
    for i in range(1, len(values) - 1):  
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            if (values[i - 1]-values[i]>=threshold) or (values[i + 1]-values[i]>=threshold):
                minima_indices.append(i)
                threshold_zlist.append(min(values[i - 1]-values[i], values[i + 1]-values[i]) )  
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            if values[i - 1]-values[i]>=threshold:
                minima_indices.append(i) 
                threshold_zlist.append(values[i - 1]-values[i])
        if len(threshold_zlist)>=100:
            last_ten = threshold_zlist#[-100:]  
            # avg = sum(last_ten) / len(last_ten)
            avg=min(last_ten)
            threshold=avg
    return minima_indices,threshold,threshold_zlist

def extract_by_html2text_db_chongdie(sub_text,model,tokenizer,threshold,language='zh') -> List[str]:   
    temp_para=sub_text

    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para
 
    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)

    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
            index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices=find_minima(first_cluster_ppl,threshold)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        # if i==0:
        #     tmp_index.append(0)
        #     tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i],split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)

    return final_chunks

def extract_by_html2text_db_nolist(sub_text,model,tokenizer,threshold,language='zh') -> List[str]:  
    temp_para=sub_text

    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para
 
    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)

    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
            index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices=find_minima(first_cluster_ppl,threshold)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)

    return final_chunks

def extract_by_html2text_db_dynamic(sub_text,model,tokenizer,threshold,threshold_zlist,language='zh') -> List[str]:  
    temp_para=sub_text 
    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para

    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)

    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
            index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices,threshold,threshold_zlist=find_minima_dynamic(first_cluster_ppl,threshold,threshold_zlist)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)
    # temp_para经过困惑度分组
    return final_chunks,threshold,threshold_zlist

def extract_by_html2text_db_dynamic_batch(sub_text,model,tokenizer,threshold,threshold_zlist,language='zh',past_key_values=None) -> List[str]:   #不重叠
    temp_para=sub_text

    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para
 
    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)


    batch_size = 4096   #6000

    total_batches = math.ceil(input_ids.shape[1] / batch_size)   
    loss=torch.tensor([], device=model.device,dtype=torch.long)
    for i in range(total_batches): 
        start=i*batch_size
        end=start+batch_size
        input_ids_tmp=input_ids[:,start:end]

        attention_mask_tmp=attention_mask[:,:end]
        input_ids_tmp = torch.cat([tokenizer(' ', return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device), input_ids_tmp],dim=-1)
        attention_mask_tmp=torch.cat([ attention_mask_tmp, torch.ones((1, i+1), device=model.device, dtype=torch.long)  ],dim=-1)
        
        size=input_ids_tmp.shape[1]
        if attention_mask_tmp.shape[1]>24576:  #72000
            past_key_values = [  
                [k[:, :, size+1: ], v[:, :, size+1: ]]  
                for k, v in past_key_values  
            ]
            attention_mask_tmp=attention_mask_tmp[:, attention_mask_tmp.shape[1]-size-past_key_values[0][0].shape[2]:]
            # print('111',attention_mask_tmp.shape,past_key_values[0][0].shape[2])
        
        loss_tmp, past_key_values = ch.get_ppl_batch( 
            input_ids_tmp,
            attention_mask_tmp,
            past_key_values=past_key_values,
            return_kv=True
        )
        loss = torch.cat([loss, loss_tmp],dim=-1)
        # print(input_ids_tmp.shape,attention_mask_tmp.shape,past_key_values[0][0].shape[2],loss.shape)
            
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[1:len_sentences[i]].mean().item())
            # index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
        index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices,threshold,threshold_zlist=find_minima_dynamic(first_cluster_ppl,threshold,threshold_zlist)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)
    # temp_para经过困惑度分组
    return final_chunks,threshold,threshold_zlist

def extract_by_html2text_db_bench(sub_text,model,tokenizer,threshold,language='zh',past_key_values=None) -> List[str]:  
    temp_para=sub_text
    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para

    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)
  
    batch_size = 8192   #6000  4096 

    total_batches = math.ceil(input_ids.shape[1] / batch_size)   
    print('111',input_ids.shape[1])
    loss=torch.tensor([], device=model.device,dtype=torch.long)
    for i in range(total_batches): 
        start=i*batch_size
        end=start+batch_size
        input_ids_tmp=input_ids[:,start:end]
        attention_mask_tmp=attention_mask[:,:end]
        input_ids_tmp = torch.cat([tokenizer(' ', return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device), input_ids_tmp],dim=-1)
        attention_mask_tmp=torch.cat([ attention_mask_tmp, torch.ones((1, i+1), device=model.device, dtype=torch.long)  ],dim=-1)
        
        size=input_ids_tmp.shape[1]
        if attention_mask_tmp.shape[1]>24576:  #72000   24576
            past_key_values = [  
                [k[:, :, size+1: ], v[:, :, size+1: ]]  
                for k, v in past_key_values  
            ]
            attention_mask_tmp=attention_mask_tmp[:, attention_mask_tmp.shape[1]-size-past_key_values[0][0].shape[2]:]
            # print('111',attention_mask_tmp.shape,past_key_values[0][0].shape[2])
        
        loss_tmp, past_key_values = ch.get_ppl_batch( 
            input_ids_tmp,
            attention_mask_tmp,
            past_key_values=past_key_values,
            return_kv=True
        )
        loss = torch.cat([loss, loss_tmp],dim=-1)
        # print(input_ids_tmp.shape,attention_mask_tmp.shape,past_key_values[0][0].shape[2],loss.shape)
            
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[1:len_sentences[i]].mean().item())
            # index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
        index+=len_sentences[i]
    # print('333',first_cluster_ppl)
    # print(first_cluster_ppl) 
    minima_indices=find_minima(first_cluster_ppl,threshold)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)

    return final_chunks
 