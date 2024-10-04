import torch
from torchmetrics.classification import BinaryAUROC

from common import get_dataloaders, get_teacher_model, get_student_model, get_Adam_optimizer, get_lr_scheduler, knowledge_distillation_loss


class Knowledge_Distillation:
    
    def __init__(self):
        
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        # get data loader 
        self.train_loader, self.validation_loader, self.test_loader = get_dataloaders()

        # get teacher model
        self.teacher_model = get_teacher_model()
        print(self.teacher_model)
        # get student model 
        self.student_model = get_student_model()
        print(self.student_model)
        # get optimizer
        self.optimizer = get_Adam_optimizer(model=self.student_model)
        
        # get_lr_scheduler
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer)
           
        self.rank = 1 
    
    def train(self):
        
        validation_max_accuracy = None
        test_max_accuracy = None
        
        self.auroc = BinaryAUROC(thresholds=5).to(self.device)

        for epoch in range(100):
            running_loss, correct, total = 0.0, 0, 0
            train_losses = []
            accuracies = []
            test_accuracies = []

            for data in self.train_loader:
                self.student_model.train()   
                self.teacher_model.eval()
                
                question, response, question_shift, response_shift, masked = data             
                
                question = question.to(self.device).long()                
                response = response.to(self.device).long()

                question_shift = question_shift.to(self.device).long()
                masked = masked.to(self.device)
                response_shift = response_shift.to(self.device).long()

                # student output
                student_outputs, _ = self.student_model(question, response, question_shift, self.device)
                              
                # teacher output
                teacher_outputs, _ = self.teacher_model(question, response, question_shift, self.rank)
                
                # total loss = student loss + distillation loss

                # true_score = torch.squeeze(response).int()
 
                student_outputs = student_outputs.type(torch.double)
                teacher_outputs = teacher_outputs.type(torch.double)
            
                total_loss = knowledge_distillation_loss(logits=student_outputs,
                                                         labels=response,
                                                         teacher_logits=teacher_outputs)
                self.optimizer.zero_grad()

                total_loss.backward()
                self.optimizer.step()
                
                true_score = response.flatten()

                if len(set(true_score.detach().cpu().numpy())) == 2:
                    train_losses.append(float(total_loss.detach().cpu().numpy()))
        
            # start validation
            with torch.no_grad():  
                for data in self.validation_loader:   
                    self.student_model.eval()
                    
                    question, response, question_shift, response_shift, masked = data

                    question = question.to(self.rank)
                    response = response.to(self.rank)
                    question_shift = question_shift.to(self.rank)
                    masked = masked.to(self.rank)
                    response_shift = response_shift.to(self.rank)

                    # student output
                    student_predict, _ = self.student_model(question, response, question_shift, self.rank) 
                    student_outputs = torch.masked_select(student_predict, masked).to(self.rank)            
                    
                    true_score = torch.masked_select(response_shift, masked).float().to(self.rank)
                    
                    if len(set(true_score.detach().cpu().numpy())) == 2:  
                        accuracy = self.auroc(student_outputs, true_score)           
                        accuracies.append(accuracy)

                    result = torch.mean(torch.tensor(train_losses))
                    accuracy_mean = torch.mean(torch.tensor(accuracies))
                    print(
                        "Validation Epoch: {}, AUC: {}, Loss Mean: {}"
                            .format(epoch, accuracy_mean, result)
                    )

                    # 정확도 첫번째
                    if validation_max_accuracy is None:               
                        validation_max_accuracy = accuracy
                        torch.save(self.student_model.state_dict(),
                                   "student_model.pth")            
                        
                    # 이전 정확도 보다 클 경우
                    if validation_max_accuracy < accuracy:
                        validation_max_accuracy = accuracy

                        print("validation_max_accuracy UPDATE")
                        torch.save(self.student_model.state_dict(),
                                   "student_model.pth") 
            # start test
            with torch.no_grad():  
                for data in self.test_loader:   

                    question, response, question_shift, response_shift, masked = data

                    question = question.to(self.rank)
                    response = response.to(self.rank)
                    question_shift = question_shift.to(self.rank)
                    masked = masked.to(self.rank)
                    response_shift = response_shift.to(self.rank)

                    # student output
                    student_predict, _ = self.student_model(question, response, question_shift, self.rank) 
                    student_outputs = torch.masked_select(student_predict, masked).to(self.rank)  

                    true_score = torch.masked_select(response_shift, masked).float().to(self.rank)
                    
                    if len(set(true_score.detach().cpu().numpy())) == 2:  
                        accuracy = self.auroc(student_outputs, true_score)         
                        test_accuracies.append(accuracy)

                    result = torch.mean(torch.tensor(train_losses))
                    test_accuracy_mean = torch.mean(torch.tensor(test_accuracies))        
                    print(
                        "Test Epoch: {}, AUC: {}"
                            .format(epoch, test_accuracy_mean)
                    )
                    print()   
                    
                    # 정확도 첫번째
                    if test_max_accuracy is None:       
                        test_max_accuracy = accuracy
                    # 이전 정확도 보다 클 경우    
                    if test_max_accuracy < accuracy:
                        test_max_accuracy = accuracy                        
                        print("test_max_accuracy UPDATE")
            
                        torch.save(self.student_model.state_dict(),
                                   "student_model.pth") 