# :mag: Project Guide

This warehouse is used to store the algorithm problems obtained from practicing Leetcode. It retains the more classic problems, including sorting, searching, and various algorithm challenges. For more information, visit the [LeetCode official website](https://leetcode.cn/problems).  

## :book: Algorithm Introduction

___1.回溯法___: 回溯法 采用试错的思想，它尝试分步的去解决一个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效的正确的解答的时候，它将取消上一步甚至是上几步的计算，再通过其它的可能的分步解答再次尝试寻找问题的答案。回溯法通常用最简单的递归方法来实现，在反复重复上述的步骤后可能出现两种情况：找到一个可能存在的正确的答案；在尝试了所有可能的分步方法后宣告该问题没有答案。

___2.深度优先搜索法___: Depth-First-Search，DFS是一种用于遍历或搜索树或图的算法。这个算法会尽可能深的搜索树的分支。当结点v的所在边都己被探寻过，搜索将回溯到发现结点v的那条边的起始结点。这一过程一直进行到已发现从源结点可达的所有结点为止。如果还存在未被发现的结点，则选择其中一个作为源结点并重复以上过程，整个进程反复进行直到所有结点都被访问为止。  

___3.广度优先搜索法___: Breadth-First-Search，BFS是一种用于遍历或搜索树或图的算法。该算法从根节点开始，沿着树的宽度遍历树的节点。如果所有节点均被访问，则算法中止。
