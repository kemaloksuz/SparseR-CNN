import torch
import pdb 

class aLRPLossv2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta_pos=1., delta_neg=1., eps=1e-5): 
        '''
        logits=torch.tensor([-1,0,1]).cuda()
        targets=torch.tensor([1,1,1]).cuda()
        regression_losses=torch.tensor([0.5,0.5,0.5]).cuda()
        '''

        classification_grads=torch.zeros(logits.shape).cuda()
        #global counter 
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta_neg
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        
        relevant_bg_logits = logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        loc_weight=torch.zeros(fg_num).cuda()
        lrp_fp=torch.zeros(fg_num).cuda()
        lrp_error=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            if delta_pos > 0:
                fg_relations_loc=torch.clamp(fg_relations/(2*delta_pos)+0.5,min=0,max=1)
            else:
                fg_relations_loc = (fg_relations >= 0).float()

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            if delta_neg > 0:
                fg_relations=torch.clamp(fg_relations/(2*delta_neg)+0.5,min=0,max=1)
                bg_relations=torch.clamp(bg_relations/(2*delta_neg)+0.5,min=0,max=1)
            else:
                bg_relations = (relevant_bg_logits > fg_logits[ii]).float()
                fg_relations = (fg_relations >= 0).float()

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank=rank_pos+FP_num
            if delta_pos == -5:
                loc_weight += (fg_relations_loc/torch.sum(fg_relations_loc))
            else:
                loc_weight += (((fg_relations >= 0).float())/torch.sum(fg_relations_loc))
                            
            #Compute precision for this example to compute classification loss 
            lrp_fp[ii]=FP_num/rank               
            lrp_loc = torch.sum(fg_relations*regression_losses)/rank_pos

            iou_relations = (regression_losses[ii] >= regression_losses)
            target_iou_relations = iou_relations * fg_relations 
            rank_pos_target = torch.sum(target_iou_relations) 
            target_lrp = torch.sum(target_iou_relations*regression_losses)/rank_pos_target
            lrp_error[ii]= -(target_lrp-(lrp_fp[ii]+lrp_loc))
            fg_grad[ii] -= lrp_error[ii]
  
            if FP_num > eps:
                relevant_bg_grad += (bg_relations*(lrp_fp[ii]/FP_num))

            fg_err_relations = (~ iou_relations) * fg_relations
            total_fg_err = torch.sum(fg_err_relations)
            if total_fg_err > eps:
                fg_grad -= (fg_err_relations*((target_lrp-lrp_loc)/total_fg_err))

        #aLRP with grad formulation fg gradient
        fg_grad /= fg_num
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= (relevant_bg_grad/fg_num)
        #print("fg total grad=", '{:7.5f}'.format(classification_grads[fg_labels].sum()), "bg total grad=", '{:7.5f}'.format(classification_grads[relevant_bg_labels].sum()))
    
        ctx.save_for_backward(classification_grads)

#        return lrp_fp.mean(), loc_weight/fg_num, torch.abs(classification_grads).sum()
#        print(lrp_fp.mean(), lrp_error.mean())
        return lrp_error.mean(), loc_weight/fg_num, classification_grads

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None, None, None

class aLRPLossv2sep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=0.5, eps=1e-5): 

        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        
        relevant_bg_logits = logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        sorting_error=torch.zeros(fg_num).cuda()
        ranking_error=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            if delta > 0:
                fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
                bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)
            else:
                bg_relations = (relevant_bg_logits > fg_logits[ii]).float()
                fg_relations = (fg_relations >= 0).float()

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank=rank_pos+FP_num
                            
            #Compute precision for this example to compute classification loss 
            ranking_error[ii]=FP_num/rank               
            lrp_loc = torch.sum(fg_relations*regression_losses)/rank_pos

            iou_relations = (regression_losses[ii] >= regression_losses)
            target_iou_relations = iou_relations * fg_relations 
            rank_pos_target = torch.sum(target_iou_relations) 
            target_lrp = torch.sum(target_iou_relations*regression_losses)/rank_pos_target
            sorting_error[ii] = lrp_loc - target_lrp 

            #Identity Update for Positive
            fg_grad[ii] -= (ranking_error[ii]+sorting_error[ii])
  
            #Identity Update for Negative 
            if FP_num > eps:
                relevant_bg_grad += (bg_relations*(ranking_error[ii]/FP_num))
            
            #Interaction with other Positives
            fg_err_relations = (~ iou_relations) * fg_relations
            total_fg_err = torch.sum(fg_err_relations)
            if total_fg_err > eps:
                fg_grad -= (fg_err_relations*((target_lrp-lrp_loc)/total_fg_err))

        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= (fg_grad/fg_num)
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= (relevant_bg_grad/fg_num)
        #print("fg total grad=", '{:7.5f}'.format(classification_grads[fg_labels].sum()), "bg total grad=", '{:7.5f}'.format(classification_grads[relevant_bg_labels].sum()))    
        ctx.save_for_backward(classification_grads)

        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, out_grad1, out_grad2):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None

class aLRPLossv1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, regression_losses, delta=1., eps=1e-5): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example to compute classification loss 
            prec[ii]=rank_pos/rank[ii]                
            #For stability, set eps to a infinitesmall value (e.g. 1e-6), then compute grads
            if FP_num > eps:   
                fg_grad[ii] = -(torch.sum(fg_relations*regression_losses)+FP_num)/rank[ii]
                relevant_bg_grad += (bg_relations*(-fg_grad[ii]/FP_num))   
                    
        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= (fg_num)
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss, rank, order

    @staticmethod
    def backward(ctx, out_grad1, out_grad2, out_grad3):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None, None, None
    
    
class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta=1.): 
        classification_grads=torch.zeros(logits.shape).cuda()
        
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        #Do not use bg with scores less than minimum fg logit
        #since changing its score does not have an effect on precision
        threshold_logit = torch.min(fg_logits)-delta

        #Get valid bg logits
        relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
        relevant_bg_logits=logits[relevant_bg_labels] 
        relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
        rank=torch.zeros(fg_num).cuda()
        prec=torch.zeros(fg_num).cuda()
        fg_grad=torch.zeros(fg_num).cuda()
        
        max_prec=0                                           
        #sort the fg logits
        order=torch.argsort(fg_logits)
        #Loops over each positive following the order
        for ii in order:
            #x_ij s as score differences with fgs
            fg_relations=fg_logits-fg_logits[ii] 
            #Apply piecewise linear function and determine relations with fgs
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            #Discard i=j in the summation in rank_pos
            fg_relations[ii]=0

            #x_ij s as score differences with bgs
            bg_relations=relevant_bg_logits-fg_logits[ii]
            #Apply piecewise linear function and determine relations with bgs
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            #Compute the rank of the example within fgs and number of bgs with larger scores
            rank_pos=1+torch.sum(fg_relations)
            FP_num=torch.sum(bg_relations)
            #Store the total since it is normalizer also for aLRP Regression error
            rank[ii]=rank_pos+FP_num
                            
            #Compute precision for this example 
            current_prec=rank_pos/rank[ii]
            
            #Compute interpolated AP and store gradients for relevant bg examples
            if (max_prec<=current_prec):
                max_prec=current_prec
                relevant_bg_grad += (bg_relations/rank[ii])
            else:
                relevant_bg_grad += (bg_relations/rank[ii])*(((1-max_prec)/(1-current_prec)))
            
            #Store fg gradients
            fg_grad[ii]=-(1-max_prec)
            prec[ii]=max_prec 

        #aLRP with grad formulation fg gradient
        classification_grads[fg_labels]= fg_grad
        #aLRP with grad formulation bg gradient
        classification_grads[relevant_bg_labels]= relevant_bg_grad 
 
        classification_grads /= fg_num
    
        cls_loss=1-prec.mean()
        ctx.save_for_backward(classification_grads)

        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None
