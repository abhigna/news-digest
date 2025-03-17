# Tech News Digest System with Integrated Evaluation Framework

## System Architecture

### 1. Data Collection
- Multiple tech news sources (Hacker News, TechCrunch, The Verge, ArsTechnica)
- Store article metadata and content for processing
- Log data collection events for debugging

### 2. Content Filtering
- Interest profile-based relevance scoring using LLM
- Selection based on relevance thresholds

### 3. Content Summarization
- LLM-based summarization of selected articles
- Key points extraction for each article

### 4. Digest Compilation
- Organize articles by topics
- Present as simple console output

## Multi-Level Evaluation Framework

### Level 1: Unit Tests (Fast & Frequent)

#### Test Definition by Feature/Scenario
- **Feature 1: Article Relevance Filtering**
  - Scenario 1.1: Highly relevant article is selected
  - Scenario 1.2: Irrelevant article is filtered out
  - Scenario 1.3: Edge case articles require nuanced filtering

- **Feature 2: Article Summarization**
  - Scenario 2.1: Summary captures main points accurately
  - Scenario 2.2: Summary maintains factual correctness
  - Scenario 2.3: Summary highlights information relevant to your interests

- **Feature 3: Digest Compilation**
  - Scenario 3.1: Articles are properly categorized
  - Scenario 3.2: Digest includes appropriate number of articles
  - Scenario 3.3: Digest presents information in readable format

#### Test Case Generation
- Generate synthetic test cases for each scenario using LLMs:
  ```
  Create 5 examples of tech articles that would be highly relevant to a software engineer 
  interested in [your specific interests]. Include title, source, date, and 2-3 paragraphs of content.
  ```

#### Assertions
Create specific assertions for each scenario, such as:
- `assert selected_articles.contains(highly_relevant_articles)`
- `assert not selected_articles.contains(irrelevant_articles)`
- `assert len(summary.split('.')) <= 5`
- `assert all(key in summary for key in article.key_points)`

#### Run Tests Frequently
- Execute these tests on every significant code/prompt change
- Track pass rates over time to monitor progress

### Level 2: Human & Model Evaluation (Weekly)

#### Trace Logging
- Implement comprehensive logging of:
  - Article full text and metadata
  - LLM reasoning during filtering and summarization
  - Generated summaries and relevance scores
  - Final digest format

#### Pass/Fail Judgments with Critiques (Critique Shadowing)
- For each test case, evaluate with binary pass/fail judgment
- Write detailed critiques explaining reasoning, such as:
  ```
  Article: "New Framework for Distributed Systems Released"
  Selected: Yes
  Summary: "Google released DistSys, a new framework for building distributed systems 
  with simplified API and improved fault tolerance. The framework supports multiple 
  languages and can be deployed on major cloud providers."
  
  Judgment: Pass
  Critique: The summary correctly captures the key information (what was released, 
  by whom, key features) in a concise format. It focuses on the technical aspects 
  most relevant to a developer while omitting marketing language from the original. 
  The information is factually accurate based on the source article.
  ```

#### Build LLM Judge Based on Your Critiques
- Create a judge prompt with your examples and critiques:
  ```
  You are evaluating a tech news digest system for a software engineer interested in [your interests].
  
  Your task is to evaluate whether article selections and summaries meet quality criteria.
  
  Here are examples of good and bad article selections and summaries with explanations:
  
  <examples>
  [Include your pass/fail examples with detailed critiques]
  </examples>
  
  For the following article and summary, write a detailed critique explaining your 
  reasoning, then provide a binary pass/fail judgment.
  ```

#### Correlation Tracking
- Track agreement between your judgments and the LLM judge
- Refine the judge prompt until you achieve >90% agreement
- Log all disagreements for further analysis

### Level 3: Long-Term Evaluation (Monthly)

#### A/B Testing Different Approaches
- Compare different filtering strategies and summarization techniques
- Measure your satisfaction with each approach over time

#### Track Metrics Over Time
- Maintain a dashboard of key metrics:
  - Pass rates by feature and scenario
  - Agreement rates between you and LLM judge
  - Error categories and their frequencies

#### User Satisfaction Metrics
- Rate each digest on a 1-10 scale for:
  - Time saved vs. manual browsing
  - Discovery of valuable content
  - Quality of information extraction

## Implementation Strategy

### 1. Set Up Logging Infrastructure
- Create a simple JSON-based logging system that records:
  - Raw input articles
  - Filtering decisions with reasoning
  - Generated summaries with LLM reasoning
  - Final digest compilation

### 2. Build Review Interface
- Create a simple console or web interface to review digests
- Include functionality to:
  - View original articles alongside summaries
  - Record pass/fail judgments with critiques
  - Track evaluation metrics over time

### 3. Test Data Generation
- Use LLMs to generate diverse test cases
- Include edge cases that test system boundaries:
  ```
  Generate 5 examples of tech articles that are borderline relevant to [your interests] 
  - articles where it's not immediately clear if they should be included or not.
  ```

### 4. Error Analysis Dashboard
- Create a simple visualization of error types:
  - Filtering errors (false positives/negatives)
  - Summary accuracy problems
  - Missing key information
  - Context misunderstandings

### 5. Continuous Improvement Process
Following Husain's "virtuous cycle":
1. Evaluate quality using tests and human review
2. Debug issues through trace analysis
3. Improve system through prompt refinement or code changes
4. Repeat