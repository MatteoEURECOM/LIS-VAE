import main
args=main.parse_args()

args.dim=32

args.if_test = False
args.AUROC=True
args.TEST = False
args.MMD= False

args.BayesianDec = True
args.BayesianEnc = False
args.m=1
main.main(args)

args.BayesianDec = True
args.BayesianEnc = False
args.m=1
main.main(args)

args.BayesianDec = False
args.BayesianEnc = False
args.m=1
main.main(args)



'''





args.if_test = False
args.TEST = False
for args.t in [1]:# [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]:
     for args.m in [1]:
         args.eps_te= 0.5
         args.eps_tr = 0.2
         args.AUROC=True
         args.MMD=False
         main.main(args)

         args.eps_te = 0.01
         args.eps_tr = 0.2
         args.AUROC = False
         args.MMD = True
         main.main(args)'''
