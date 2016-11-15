#execfile('RIFModel.py')
execfile('RIFModelFreq_changeall.py')
#execfile('RIFModelFreq_change50.py')

#pahse 1 - learning
mapping = {
    'DRINKS': ['VODKA', 'BOURBON', 'RUM','ALE','GIN','WHISKEY'],
    'WEAPONS': ['SWORD','RIFLE','TANK','BOMB','PISTOL','CLUB'],
    'FISH': ['CATFISH','HERRING','TROUT','BLUGILL','FLOUNDER','GUPPY'],
    'FRUITS' : ['TOMATO','STRAWBERRY','BANANA','ORANGE','LEMON','PINEAPPLE'],
    'PROFESSIONS' : ['ENGINEER','ACCOUNTANT','DENTIST','NURSE','PLUMBER','FARMER'],
    'METALS' : ['IRON','ALUMINUM','NICKEL','SILVER','BRASS','GOLD'],
    'TREES' : ['BIRCH','HICKORY','DOGWOOD','ELM','SPRUCE','REDWOOD'],
    'INSECTS' : ['BEETLE','ROACH','HORNET','FLY','MOSQUITO','GRASSHOPPER'],
}
#mapping = {
#    'DRINKS': ['VODKA', 'BOURBON', 'RUM','ALE','GIN','WHISKEY'],
#    'WEAPONS': ['SWORD','RIFLE','TANK','BOMB','PISTOL','CLUB'],
#    }





Data=np.array([0,0,0,0])
Num_of_Ss=1
for i in range(0,Num_of_Ss):
    m = RIFModelFreq(mapping, D_category=16, D_items=64, threshold=0.4, learning_rate=1e-4)

    categories=mapping.keys()
    rnd.shuffle(categories)
    Practiced_Categories=categories[:4] #choose randomly 4 categoris
    UnPracticed_Categories=categories[4:]

    #choose items
    Rp_Minus_low_freq={}
    Rp_Minus_high_freq={}
    Rp_Plus={}
    nRP=[] #lets see if we need to define it..
    for pc in Practiced_Categories:
        #rnd.shuffle(mapping[pc])
        Rp_Minus_low_freq[pc]=mapping[pc][:2]
        Rp_Minus_high_freq[pc]=mapping[pc][2:4]
        Rp_Plus[pc]=mapping[pc][4:]

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
            m.practice_reverse(cat,item,'category')

    #phase 4 - test
    Rp_Plus_Test=[]
    for pc in Rp_Plus:
        Rp_Plus_Test.append(m.test(pc,items=Rp_Plus[pc]))
  
    Rp_Minus_low_Test=[]
    for pc in Rp_Minus_low_freq:
        Rp_Minus_low_Test.append(m.test(pc,items=Rp_Minus_low_freq[pc]))
    Rp_Minus_high_Test=[]
    for pc in Rp_Minus_high_freq:
        Rp_Minus_high_Test.append(m.test(pc,items=Rp_Minus_high_freq[pc]))


    nRp_Test=[]
    for pc in UnPracticed_Categories:
        nRp_Test.append(m.test(pc))

 
    Data=np.vstack((Data,[np.mean(Rp_Plus_Test),np.mean(nRp_Test),np.mean(Rp_Minus_low_Test),np.mean(Rp_Minus_high_Test)]))

Data=Data[1:] #removes the [0,0,0] line (1st line)
Rpp=Data[:,0] #+
nRp=Data[:,1] #non pracctice
Rpn_low=Data[:,2] #-low
Rpn_high=Data[:,3] #-high

np.savetxt('Freq.csv', Data, delimiter=',')

Means=[np.mean(Rpp),np.mean(nRp),np.mean(Rpn_low),np.mean(Rpn_high)]
print Means
print "Rpp"+str(np.mean(Rpp))+" nRp"+str(np.mean(nRp))+" Rpn_low"+str(np.mean(Rpn_low))+" Rpn_high"+str(np.mean(Rpn_high)) 
print "Rpp-nRp"
print scipy.stats.ttest_rel(Rpp,nRp)
print "nRp-Rpn_low"
print scipy.stats.ttest_rel(nRp,Rpn_low)
print "nRp-Rpn_high"
print scipy.stats.ttest_rel(nRp,Rpn_high)

