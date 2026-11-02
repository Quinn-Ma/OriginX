"""Synthetic-only CPU tests; no experiment data or private dependencies."""
import copy,csv,tempfile,unittest
from pathlib import Path
import reproduce_memory as R

def synthetic():
    return [dict(task=t,init=i,baseline=(j%4==0),memory=(j%2==0),reset=(j%4==0))
            for j,(t,i) in enumerate((t,i) for t in R.TASKS for i in R.INITIAL_STATES)]
def write(path,rows,columns=R.COLUMNS):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
class Reproduction(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'synthetic.csv';write(self.path,synthetic())
    def tearDown(self):self.temp.cleanup()
    def test_complete_counts_paired_statistics_and_reference(self):
        result=R.summarize(R.load_pairs(self.path))
        self.assertEqual(result['successes'],dict(baseline=5,memory=10,reset=5))
        for name in ('memory_vs_baseline','memory_vs_reset'):
            c=result['contrasts'][name];self.assertEqual((c['rescues'],c['regressions']),(5,0));self.assertEqual(c['difference_pp'],25.)
            self.assertEqual(c['mcnemar_exact_two_sided_p'],.0625);self.assertEqual(c['holm_adjusted_p'],.125)
        reference=copy.deepcopy(result);reference['independent_recount_matches']=True
        R.compare_reference(result,reference)
        reference['contrasts']['memory_vs_reset']['difference_pp']=30.
        with self.assertRaises(ValueError):R.compare_reference(result,reference)
    def test_missing_duplicate_and_unknown_scene_rejected(self):
        for rows in (synthetic()[:-1],synthetic()[:-1]+[synthetic()[0]],synthetic()+[dict(task=9,init=40,baseline=False,memory=False,reset=False)]):
            write(self.path,rows)
            with self.assertRaises(ValueError):R.load_pairs(self.path)
    def test_non_boolean_values_rejected(self):
        for bad in ('1','0','yes','','NaN','None'):
            rows=synthetic();rows[0]['memory']=bad;write(self.path,rows)
            with self.assertRaises(ValueError,msg=bad):R.load_pairs(self.path)
        pairs=synthetic();pairs[0]['memory']=1
        with self.assertRaises(ValueError):R.summarize(pairs)
    def test_duplicate_header_extra_and_missing_fields_rejected(self):
        for text in ('task,init,baseline,memory,memory\n','task,init,baseline,memory,reset\n0,40,True,False\n','task,init,baseline,memory,reset\n0,40,True,False,True,extra\n'):
            self.path.write_text(text)
            with self.assertRaises(ValueError):R.load_pairs(self.path)
    def test_row_order_irrelevant_and_zero_discordance(self):
        rows=list(reversed(synthetic()));write(self.path,rows);self.assertEqual(R.load_pairs(self.path),synthetic())
        for r in rows:r.update(baseline=False,memory=False,reset=False)
        write(self.path,rows);result=R.summarize(R.load_pairs(self.path))
        for c in result['contrasts'].values():
            self.assertEqual(c['ci95_unadjusted_pp'],[0.,0.]);self.assertEqual(c['mcnemar_exact_two_sided_p'],1.);self.assertIsNone(c['relative_success_change_percent'])
if __name__=='__main__':unittest.main(verbosity=2)
