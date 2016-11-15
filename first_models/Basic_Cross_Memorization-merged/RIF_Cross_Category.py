#execfile('RIFModel.py')
execfile('RIFModelRev2.py')
       
#pahse 1 - learning
mapping={
    'Rp1_cue': ['Q11','Q12','Q13','Q14','Q15','Q16'],
    'NRp1_cue':['Q21','Q22','Q23','Q14','Q15','Q16'],
    'Rp2_cue':['Q31','Q32','Q33','Q34','Q35','Q36'],
    'NRp2_cue':['Q41','Q42','Q43','Q34','Q35','Q36'],
    'NRp3':['Q51','Q52','Q53','Q54','Q55','Q56'],
    'NRp4':['Q61','Q62','Q63','Q64','Q65','Q66'],
} #the test is whether the retrical of Rp1_cue-Q14 will supress Rp2_cue-Q11
    

Data=np.array([0,0,0,0,0,0])
Num_of_Ss=1
for i in range(0,Num_of_Ss):
    m = RIFModelRev(mapping, learning_rate=1e-5,DimVocab=256)
    
    Practiced_Categories=['Rp1_cue','Rp2_cue'] 
    Induced_Supression=['NRp1_cue','NRp2_cue']#it is unpracticed
    UnPracticed_Categories=['NRp3','NRp4']

    #choose items
    Rp_Plus={}
    Rp_Minus={}
    nRP=[] #lets see if we need to define it..
    for pc in Practiced_Categories:
        Rp_Plus[pc]=mapping[pc][:3]
        Rp_Minus[pc]=mapping[pc][3:]

    Induced_Supression_Supressed={} #the would be supressed items 
    Induced_Supression_Control={} #the control
    for pc in Induced_Supression:
        Induced_Supression_Supressed[pc]=mapping[pc][3:] #its the reversed order than the last
        Induced_Supression_Control[pc]=mapping[pc][:3]
    
    
    #Pre testing
    #Pre_Activation_Practiced_Categories=[]
    #for pc in Practiced_Categories:
    #    Pre_Activation_Practiced_Categories.append(m.test(pc))

    #Pre_Activation_UnPracticed_Categories=[]
    #for pc in UnPracticed_Categories:
    #    Pre_Activation_UnPracticed_Categories.append(m.test(pc))

    #phase 2 - practicing : RP+
    for cat in Rp_Plus:
        for item in Rp_Plus[cat]:
            m.practice_reverse(cat,item,'category')
    
    #phase 4 - test
    Rp_Plus_Test=[]
    for pc in Rp_Plus:
        Rp_Plus_Test.append(m.test(pc,items=Rp_Plus[pc]))
    
    Rp_Minus_Test=[]
    for pc in Rp_Minus:
        Rp_Minus_Test.append(m.test(pc,items=Rp_Minus[pc]))

    Induced_Supression_Supressed_Test=[]
    for pc in Induced_Supression_Supressed:
        Induced_Supression_Supressed_Test.append(m.test(pc,items=Induced_Supression_Supressed[pc]))

    Induced_Supression_Control_Test=[]
    for pc in Induced_Supression_Control:
        Induced_Supression_Control_Test.append(m.test(pc,items=Induced_Supression_Control[pc]))

    nRp_Test=[]
    for pc in UnPracticed_Categories:
        nRp_Test.append(m.test(pc))

    print "Rp+:"+str(np.mean(Rp_Plus_Test))+" "+"nRp:"+str(np.mean(nRp_Test))+" "+"Rp-:"+str(np.mean(Rp_Minus_Test))

    Data=np.vstack((Data,[np.mean(Rp_Plus_Test),np.mean(nRp_Test),np.mean(Rp_Minus_Test),0,np.mean(Induced_Supression_Control_Test),np.mean(Induced_Supression_Supressed_Test)]))

Data=Data[1:] #removes the [0,0,0] line (1st line)

Rpp=Data[:,0] #+
nRp=Data[:,1] #non pracctice
Rpn=Data[:,2] #-

np.savetxt('RIF_Cross_Category.csv', Data, delimiter=',')

Means=[np.mean(Rpp),np.mean(nRp),np.mean(Rpn)]
SDs=[np.std(Rpp),np.std(nRp),np.std(Rpn)]
print scipy.stats.ttest_rel(Rpp,nRp)
print scipy.stats.ttest_rel(nRp,Rpn)
