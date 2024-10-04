import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from model.sakt import SAKT, Student_SAKT
from data_loader.dataloader import Test


def match_sequence_length(
        user_list: list, 
        question_sequences: list,
        response_sequences: list,
        sequence_length: int,
        padding_value=-1
):
    """Function that matches the length of question_sequences and response_sequences to the length of sequence_length.

    Args:
        question_sequences: A list of question solutions for each student.
        response_sequences: A list of questions and answers for each student.
        sequence_length: Length of sequence.
        padding_value: Value of padding.

    Returns:
        length-matched parameters.

    Note:
        Return detail.

        - proc_question_sequences : length-matched question_sequences.
        - proc_response_sequences : length-matched response_sequences.

    Examples:
        >>> match_sequence_length(question_sequences=[[67 18 67 18 67 18 18 67 67 18 18 35 18 32 18]...],
        >>>                       response_sequence=[[0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0]...],
        >>>                       sequence_length=50,
        >>>                       padding_value=-1)
        ([[67 18 67 18 67 18 18 67 67 18 18 35 18 32 18 -1 -1 -1 ... -1 -1 -1]...],
        [[0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 -1 -1 -1 ... -1 -1 -1]...])
    """
    proc_question_sequences = []
    proc_response_sequences = []
    proc_user_list = []
    
    for user, question_sequence, response_sequence in zip(user_list, question_sequences, response_sequences):

        i = 0

        while i + sequence_length + 1 < len(question_sequence):
            proc_user_list.append(user)
            proc_question_sequences.append(question_sequence[i:i + sequence_length + 1])
            proc_response_sequences.append(response_sequence[i:i + sequence_length + 1])
            i += sequence_length + 1
        
        proc_user_list.append(user)
        proc_question_sequences.append(
            np.concatenate(
                [
                    question_sequence[i:],
                    np.array([padding_value] * (i + sequence_length + 1 - len(question_sequence)))
                ]
            )
        )

        proc_response_sequences.append(
            np.concatenate(
                [
                    response_sequence[i:],
                    np.array([padding_value] * (i + sequence_length + 1 - len(question_sequence)))
                ]
            )
        )

    return proc_user_list, proc_question_sequences, proc_response_sequences


def get_dataloaders():
    # load dataset
    dataset = Test()

    # data split
    # 총 데이터 수   
    dataset_size = len(dataset.questions.index)

    # 훈련 데이터 수
    train_size = int(dataset_size * float(0.8))

    # 검증 데이터 수
    validation_size = int(dataset_size * 0.1)  

    # 데스트 데이터 수 (일반화 성능 측정)
    test_size = dataset_size - train_size - validation_size    

    # random_split 활용    
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size])

    # 훈련 데이터 로더
    train_loader = DataLoader(        
        dataset=train_dataset,
        num_workers=8,   
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=1024
    )
 
    # 검증 데이터 로더는
    validation_loader = DataLoader(
        dataset=validation_dataset,
        num_workers=2, 
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=validation_size
    )
   
    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=1,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=test_size
    )

    return train_loader, validation_loader, test_loader


def get_teacher_model():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model_config = np.load("./data/model_config.npy", allow_pickle=True).item()
      
    # set model
    teacher_model =  SAKT(
        number_questions=int(model_config["number_questions"]),
        n=int(model_config["n"]),
        d=int(model_config["d"]),
        number_attention_heads=int(model_config["number_attention_heads"]),
        dropout=float(model_config["dropout"])
    ).to(device)

    ckeckpoint = torch.load("./data/model.pth", weights_only=True)
    teacher_model.load_state_dict(ckeckpoint)    
    
    return teacher_model


def get_student_model():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model_config = np.load("./data/model_config.npy", allow_pickle=True).item()

    # set model
    test = int(model_config["number_questions"])
    student_model =  Student_SAKT(
        number_questions=test,
        n=int(model_config["n"]),
        d=int(model_config["d"]),
        number_attention_heads=int(model_config["number_attention_heads"]),
        dropout=float(model_config["dropout"])
    ).to(device)
    
    return student_model


def get_Adam_optimizer(model):
    return Adam(model.parameters(), float(0.001))


def get_lr_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)


def knowledge_distillation_loss(logits, labels, teacher_logits):
        alpha = 0.1
        T = 10
        true_score = torch.squeeze(labels)
        true_score = torch.tensor(true_score, dtype=torch.double)
        student_loss = binary_cross_entropy(input=logits, target=true_score)
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits/T, dim=0), F.softmax(teacher_logits/T, dim=0)) * (T * T)
        total_loss =  alpha*student_loss + (1-alpha)*distillation_loss

        return total_loss
    
    
def collate_fn(
    batch,
    padding_value=-1
):
    """The collate function for torch.utils.data.DataLoader

    Args:
        batch: data batch.
        padding_value: Value of padding.

    Returns:
        Dataloader elements for model training.

    Note:
        Return detail.

        - question_sequences: the question(KC) sequences.
            - question_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - response_sequences: the response sequences.
            - response_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - question_shift_sequences: the question(KC) sequences which were shifted one step to the right.
            - question_shift_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - response_shift_sequences: the response sequences which were shifted one step to the right.
            - response_shift_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - mask_sequences: the mask sequences indicating where the padded entry.
            - mask_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].
    """
    question_sequences = []
    response_sequences = []
    question_shift_sequences = []
    response_shift_sequences = []

    for q_seq, r_seq in batch:        
        question_sequences.append(q_seq[:-1].clone().detach())
        response_sequences.append(r_seq[:-1].clone().detach())
        question_shift_sequences.append(q_seq[1:].clone().detach())
        response_shift_sequences.append(r_seq[1:].clone().detach())

    question_sequences = pad_sequence(
        question_sequences, batch_first=True, padding_value=padding_value
    )
    response_sequences = pad_sequence(
        response_sequences, batch_first=True, padding_value=padding_value
    )
    question_shift_sequences = pad_sequence(
        question_shift_sequences, batch_first=True, padding_value=padding_value
    )
    response_shift_sequences = pad_sequence(
        response_shift_sequences, batch_first=True, padding_value=padding_value
    )

    mask_sequences = (question_sequences != padding_value) * (question_shift_sequences != padding_value)

    question_sequences, response_sequences, question_shift_sequences, response_shift_sequences = \
        question_sequences * mask_sequences, response_sequences * mask_sequences, question_shift_sequences * mask_sequences, \
        response_shift_sequences * mask_sequences

    return question_sequences, response_sequences, question_shift_sequences, response_shift_sequences, mask_sequences