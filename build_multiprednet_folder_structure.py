import os

multiprednet_root = os.getcwd()

print("Building MultiPredNet folder structure from {}".format(multiprednet_root))

if not os.path.exists('datasets'):
    os.mkdir('datasets')
os.chdir('datasets')

if not os.path.exists('trainingset'):
    os.mkdir('trainingset')
if not os.path.exists('testset1'):
    os.mkdir('testset1')
if not os.path.exists('testset2'):
    os.mkdir('testset2')
if not os.path.exists('testset3'):
    os.mkdir('testset3')
if not os.path.exists('testset4'):
    os.mkdir('testset4')

os.chdir(multiprednet_root)

if not os.path.exists('representations'):
    os.mkdir('representations')
os.chdir('representations')

if not os.path.exists('testset1'):
    os.mkdir('testset1')
os.chdir('testset1')
if not os.path.exists('blind'):
    os.mkdir('blind')
if not os.path.exists('numb'):
    os.mkdir('numb')
if not os.path.exists('unimpaired'):
    os.mkdir('unimpaired')
os.chdir(os.path.pardir)

if not os.path.exists('testset2'):
    os.mkdir('testset2')
os.chdir('testset2')
if not os.path.exists('blind'):
    os.mkdir('blind')
if not os.path.exists('numb'):
    os.mkdir('numb')
if not os.path.exists('unimpaired'):
    os.mkdir('unimpaired')
os.chdir(os.path.pardir)

if not os.path.exists('testset3'):
    os.mkdir('testset3')
os.chdir('testset3')
if not os.path.exists('blind'):
    os.mkdir('blind')
if not os.path.exists('numb'):
    os.mkdir('numb')
if not os.path.exists('unimpaired'):
    os.mkdir('unimpaired')
os.chdir(os.path.pardir)

if not os.path.exists('testset4'):
    os.mkdir('testset4')
os.chdir('testset4')
if not os.path.exists('blind'):
    os.mkdir('blind')
if not os.path.exists('numb'):
    os.mkdir('numb')
if not os.path.exists('unimpaired'):
    os.mkdir('unimpaired')

os.chdir(multiprednet_root)