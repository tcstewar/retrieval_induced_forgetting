#execfile('RIFModel.py')
execfile('RIFModelPhase1.py')

#pahse 1 - learning
mapping = {
    'DRINKS': ['VODKA', 'BOURBON', 'RUM','ALE','GIN','WHISKEY'],
    'WEAPONS': ['SWORD','RIFLE','TANK','BOMB','PISTOL','CLUB'],
    'FISH': ['CATFISH','HERRING','TROUT','BLUGILL','FLOUNDER','GUPPY'],
    'FRUITS' : ['TOMATO','STRAWBERRY','BANANA','ORANGE','LEMON','PINEAPPLE'],
    'PROFESSIONS' : ['ENGINEER','ACCOUNTANT','DENTIST','NURSE','PLUMBER','FARMER'],
    'METALS' : ['IRON','ALUMINUM','NICKEL','SILVER','BRASS','GOLD'],
    'TREES' : ['BIRCH','HICKORY','DOGWOOD','ELM','SPRUCE','REDWOOD'],
    'INSECTS' : ['BEETLE','ROACH','HORNET','FLY','MOSQUITO','GRASSHOPPER']    
}


Data=np.array([0,0,0])
Num_of_Ss=30
for i in range(0,Num_of_Ss):
    m = RIFModelPhase1(mapping, D_category=16, D_items=64, threshold=0.4, learning_rate=1e-4)


    #Phase 1
    num_categories=len(mapping.keys())
    num_items=len(mapping[mapping.keys()[0]]) #easy to covert to different items within category
    #learn_order=[[mapping.keys()[i],mapping[mapping.keys()[i][j]]] for i,j in cat_ind,item_ind]
    learn_order=[]
    for i in range(num_categories):
        for j in range(num_items):
            learn_order.append([mapping.keys()[i],mapping[mapping.keys()[i]][j]])
    rnd.shuffle(learn_order)

    num_iterations=5
    for it in range(num_iterations):
        for cat,item in learn_order:
            m.memorize(cat,item,0.5)
        rnd.shuffle(learn_order)
    
    categories=mapping.keys()
    rnd.shuffle(categories)
    Practiced_Categories=categories[:4] #choose randomly 4 categoris
    UnPracticed_Categories=categories[4:]

    #choose items
    Rp_Plus={}
    Rp_Minus={}
    nRP=[] #lets see if we need to define it..
    for pc in Practiced_Categories:
        rnd.shuffle(mapping[pc])
        Rp_Plus[pc]=mapping[pc][:3]
        Rp_Minus[pc]=mapping[pc][3:]

    #Pre testing
    Pre_Activation_Practiced_Categories=[]
    for pc in Practiced_Categories:
        Pre_Activation_Practiced_Categories.append(m.test(pc))

    Pre_Activation_UnPracticed_Categories=[]
    for pc in UnPracticed_Categories:
        Pre_Activation_UnPracticed_Categories.append(m.test(pc))

    #phase 2 - practicing : RP+
    for cat in Rp_Plus:
        for item in Rp_Plus[cat]:
            #m.practice(cat,item)
            m.practice_reverse(cat,item,'category')

    #phase 4 - test
    Rp_Plus_Test=[]
    for pc in Rp_Plus:
        Rp_Plus_Test.append(m.test(pc,items=Rp_Plus[pc]))
  
    Rp_Minus_Test=[]
    for pc in Rp_Minus:
        Rp_Minus_Test.append(m.test(pc,items=Rp_Minus[pc]))

    nRp_Test=[]
    for pc in UnPracticed_Categories:
        nRp_Test.append(m.test(pc))

 
    Data=np.vstack((Data,[np.mean(Rp_Plus_Test),np.mean(nRp_Test),np.mean(Rp_Minus_Test)]))

Data=Data[1:] #removes the [0,0,0] line (1st line)

Rpp=Data[:,0] #+
nRp=Data[:,1] #non pracctice
Rpn=Data[:,2] #-

np.savetxt('Phase1.csv', Data, delimiter=',')

Means=[np.mean(Rpp),np.mean(nRp),np.mean(Rpn)]
print Means
SDs=[np.std(Rpp),np.std(nRp),np.std(Rpn)]
print scipy.stats.ttest_rel(Rpp,nRp)
print scipy.stats.ttest_rel(nRp,Rpn)

