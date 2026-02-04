import joblib, pickle, sys
with open('no_dominant_m2_24h_nihss_cpu.pkl', 'rb') as f:
    unpickler = pickle.Unpickler(f)
    while True:
        try:
            unpickler.load()
        except EOFError:
            break
        except Exception as e:
            if 'tabpfn' in str(e):
                print(f'🎯 YOUR EXACT VERSION NEEDED: {e}')
                sys.exit(0)
print('No tabpfn info found')
