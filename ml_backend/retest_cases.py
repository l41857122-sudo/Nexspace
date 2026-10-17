import sys
import json
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from run_phase1_visual_validation import VisualValidator

def main():
    validator = VisualValidator()
    target_ids = {'115', '137', '163', '207', '208', '209', '210'}
    items = [it for it in validator.manifest if it['id'] in target_ids]
    
    print(f"Retesting {len(items)} cases...")
    retest_results = []
    for it in items:
        res = validator.process_case(it)
        retest_results.append(res)
        print(f"#{res['id']} [{res['status']}] {res['query'][:35]} -> {res.get('scene_id', 'N/A')} ({res.get('reason', '')[:50]})")

    return retest_results

if __name__ == '__main__':
    main()
