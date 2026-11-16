1. đầu tiên, u need to tìm một cái practical hoặc theoretical problem chưa được (fully) giải quyết.
ví dụ trong comsci thì có climate modeling là một practical problem chưa được (fully) giải quyết, P vs. NP là một theoretical problem chưa được solved.

hint: how to choose a problem quick?
warning: cái này hơi bias về data analysis. =))
chọn một cái field mình đang quan tâm. ví dụ, financial trading, high-energy physics, etc.
sau đó xác định loại data mà field đó sử dụng nhiều.
với financial trading thì là time-series data.
với high-energy physics thì thường là spatio-temporal data.
với climate modeling, protein modeling, etc. thì sẽ dùng các loại data khác nhau.
note: trong một field thường dùng nhiều loại data khác nhau nên chọn 1 cái thường thấy là được.
xác định được data of interest <D> rồi thì ghép <D> + "analysis". ví dụ <D> là time-series thì thành time-series analsis.
sau đó xác định các task khi làm analysis về loại data đó.
ví dụ trong time-series analysis thì có các task:
- time-series forecasting
- time-series anomaly detection
- time-series classification
- time-series missing values imputation
- time-series clustering
- synthetic time-series generation
- etc.


2. systematic literature review.
tìm survey, review article trong vòng 5 năm gần đây về <problem> hoặc <task> (từ bước 1) để biết được what solved?, what unsolved?, what are practical challenges remaining?, what are research gaps remain underexplored?, what are research questions, directions suggested next?. nếu về comsci thì nên tìm các cách khác nhau để formulate problem.
problem formulation có thể viết theo cấu trúc IPO model:
+ input: scalars, vectors, or tensors? hay data structures nào đó như trees, graphs, v.v.
+ process: which families of algorithms? which families of functions?
+ output: scalars, vectors, or tensors? hay data structures nào đó như trees, graphs, v.v.
yeah, its pure math...

how to find (maybe) high-quality surveys and papers? cái này cũng dài nên chắc để sau. lol.

3. sau khi đọc xong vài ba bài review articles với surveys thì chắc chắn là mình sẽ biết được sơ sơ problem này có những cách formulate nào, các challenge nào chưa được fully solved, có các dataset nào uy tín.

lúc này có thể chọn hoặc tự đặt ra tối đa 3 practical challenges hay research gaps để nghiên cứu sâu.
thường thì một practical challenge sẽ có ít nhất một research gap tương ứng. at this time, u should have in the pocket your research questions...

how to find good datasets? cái này cũng dài nên chắc để sau. =))))

4. time to build foundation. read foundational papers of each research direction targeting a specific high-level research question.

5. read state-of-the-art (SOTA) papers.