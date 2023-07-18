// public class Solution {
//     List<Integer> helper(int n,long k,int start){
//         List<Integer> nums = new ArrayList<>();
//         long fact = 1L;
//         for(int i = start;i <= (start + n - 1);i++){
//             if(i < (start + n - 1)) fact *= (i - start + 1);
//             nums.add(i);
//         }
//         k = k - 1;
//         List<Integer> ans = new ArrayList<>();
//         for(;;){
//             ans.add(nums.get((int)(k / fact)));
//             nums.remove((int)(k / fact));
//             if(nums.size() == 0) break;
//             k = k % fact;
//             fact /= nums.size();
//         }
//         return ans;
//     }
//     public int[] findPerm(int n, long k) {
//         if(n <= 20){
//             List<Integer> tmp = helper(n,k,1);
//             int ans[] = new int[n];
//             for(int i = 0;i < n;i++){
//                 ans[i] = tmp.get(i);
//             }
//             return ans;
//         }
//         int ans[] = new int[n];
//         int i = 0;
//         for(;i < n - 20;i++){
//             ans[i] = i + 1;
//         }
//         List<Integer> tmp = helper(20,k,n - 20 + 1);
//         for(;i < n;i++){
//             ans[i] = tmp.get(i - n + 20);
//         }
//         return ans;
//     }
// }