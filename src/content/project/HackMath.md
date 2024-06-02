---
title: "HackMath"
description: "Project for building an interactive AI Math education software"
pubDate: "May 2 2024"
heroImage: "https://i.imgur.com/DHxYMHI.pngp"
badge: "Design"
tags: ["UXUI", "Hackathon"]
---

# Background

> After COVID, students' learning gap in math has increased significantly.

National Center for Education Statistics shows that a meager **74 percent** of eighth graders show no math proficiency, increasing from 66 percent in 2019(Mervosh, 2022). Its research shows that the average student in American K-12 schools is **five months** behind in math and four months behind in reading.

However, current systematic problems prevent students from getting students back on track with math education.

> High Average Learning gap 1.5 years behind(Harvard, 2023)

> High Private Math Tutoring costs $40 per hour(Tutors.com, 2024).

> High Math Teacher to Students Ratio 1 teacher: 35.5 students[1]

This problem has no outstanding solution because of its high private math tutoring cost, high math-teacher-to-students ratio, and high learning gap.

It motivated our team to build a math education platform that provides personalized AI education to students by analyzing its learning gap.

# What is HackMath

**HackMath** is an AI platform where students can stay on track with math through customized Learning.

**1. Individual Assessment**

When students first join HackMath, they take an assessment quiz. This quiz helps the AI understand the student's current level of knowledge and areas of strength and weakness in Math.

**2. Tailored Lessons**

Based on the results of the assessment quiz, the AI customizes the learning content to match the student's specific needs. This means the lesson plan is not one-size-fits-all; it adapts to the student's learning pace and understanding.

**3. Interactive Learning Experience**

The AI tutor, named Dr. Ham, engages students through interactive lessons that make learning Math fun and engaging. This interactive component keeps students motivated and enhances their learning experience.

**4. Continuous Analysis and Adaptation**

The AI continually analyzes their performance and learning gaps as the student progresses through the lessons. This ongoing analysis allows the AI to adjust the curriculum dynamically, ensuring that the student is constantly challenged just enough to learn effectively without feeling overwhelmed.

**5. Feedback Through Analytics**

HackMath includes an analytics tab that provides feedback on the student's progress. This feature allows students(and potentially their teachers or parents) to see tangible proof of improvement and areas that need more attention.

**6. Categories of Learning**
The AI Customized Learning in HackMath is divided into three categories:

- Conceptual: Understanding the underlying concepts of Math topics.
- Procedural: Learning how to apply these concepts through mathematical procedures or steps.
- Application: Applying learned concepts and procedures to solve real-world problems.

#What it does
###Practical Implementation
Here are visual examples demonstrating how lessons are presented to students on the HackMath platform:

- **Lesson Prompt Example**
  Shows how a typical lesson is introduced to the student, setting the stage for Learning.

![Prompt Example](https://i.imgur.com/wURaz3w.png)

- **Prompt Output Example**
  Illustrates the output generated from the initial prompt, showing how the system processes and responds to the student's inputs. This output helps the student understand the procedure or concept being taught.

![Prompt result](https://i.imgur.com/Xv3tYkf.png)

Through these methods, HackMath's AI Customized Learning not only makes math education more accessible and engaging but also ensures it is highly effective by adapting to the individual learning style and pace of each student. This tailored approach helps build confidence and proficiency in mathematics among learners of all levels.

**Disclaimer**

_Currently, this is in the experimental/development phase and not yet live. We are having issues with Vercel AI SDK (Read more on Challenges). We are confident that the Generative UI can be done ( you can see https://gemini.vercel.ai/ as an example). This would give Dr. Ham superpowers and more interaction when Dr. Ham has access to students' learning data. We further describe this on challenges we faced_

This is how it will be visualized:

###**Interactive Lessons**

This visualization demonstrates the interactive nature of the lessons, where concepts are introduced through engaging, dynamic content that encourages active participation.
![visualized 1](https://i.imgur.com/vLT0GpH.png)

###**Interactive Questions**

This image shows the interactive buttons used during lessons, which allow students to answer questions, make selections, and navigate through different parts of the lesson, enhancing their learning experience through engagement.
![visualized 2](https://i.imgur.com/BAJAsnk.png)

###**Ask a question to Dr.Ham**
Here, students can interact directly with the AI tutor, Dr. Ham. This feature enables them to ask specific questions about the lesson, receive clarifications in real time, and further personalize their learning path.
![visualized 3](https://i.imgur.com/4vkmwDZ.png)

## AI chatbot - Dr. Ham

The AI chatbot is aware of **user's performance and her previous history of math lessons**. By analyzing the pattern of students' performance, interact with students in a way students can be motivated and provide proper advice.
![Chatbot](https://i.imgur.com/uUh9tJa.png)

_Disclaimer_
We are currently not using Rag, but we plan to use Pinecone and Gemini to answer based on the user's context (Learnings, grades, etc).

Utilizing Retrieval-Augmented Generation(RAG) and User Analytics(continue after this paragraph), Dr.Ham will provide proper responses and math resources. This is a basic structure of the Retrieval Augmented Generation of HackMath.
![RAG](https://i.imgur.com/1OPSSfh.png)

##User Analytics

> "There is a positive relationship between immediate feedback and mathematics learning achievement" (Research Gate).

User Analytics gives overall achievement matrices and provides a box of AI analytics feedback where students can find written feedback. We are planning on Implementing learning analytics with advanced education performance analytics and building a foundation for Retrieval-Augmented Generation(RAG) in the future. This is a basic structure of analytics flow.
![User Analytics](https://i.imgur.com/i3SMms9.png)

# UXUI

> [Figma](https://www.figma.com/file/s8L3Tktpd5ImhwqTVN2FYv/Gemini-Hack--HackMath?)

![figma](https://i.imgur.com/EkYvbFM.png)

Access our complete design suite in Figma, featuring the overall concept along with tailored versions for iOS and desktop. This file includes comprehensive layouts and interfaces that showcase our approach to user-centric design.

## Design Guide

> [Behance](https://www.behance.net/gallery/197537847/Google-AI-Hack-HackMath?)

![Behance](https://i.imgur.com/OMzsG1Z.png)
Check out our detailed design guide, featuring our project's design concept and creative process, where we explore everything from the initial sketches to the polished final product. Click the link to see the innovative designs that brought our ideas to life!

## User Persona

![UX/UIPersona](https://i.imgur.com/9K4nRof.png)

## User Flow

![UserFlow](https://i.imgur.com/2WUDXN3.png)
_User-level test evaluation flow_
![UserFlow](https://i.imgur.com/1JjcCSf.png)
_Overall user flow_
_This diagram maps out the user flow for an educational platform, detailing pathways from onboarding to interactive and AI-enhanced educational content, including courses, games, and personal analytics. Error nodes highlight potential system issues during sign-up and interactive sessions, while additional features like a shop and rank system aim to enhance user engagement and Learning._

# How we built it

1.  Creating a user-friendly platform that makes advanced AI accessible to a broader range of educators.
2.  Successfully integrating Gemini AI APIs to provide a seamless experience.
3.  Developing a responsive and interactive user interface that enhances the user experience.

Here is how it works:
**HackMath** is built by two students. Montek is a full-time computer information system student and full-time full-stack developer proficient in AI. Min, Business Data Analytics student, UI/UX designer, and Economics researcher.

**HackMath** is designed by HackMath and is created through a typical, albeit slightly rushed, design process. We conducted market research, created user personas, decided user flow, and designed high-fidelity prototypes.

We started with a basic [Nextjs app](https://github.com/Montekkundan/hackmath-test), but as we were implementing more features as ideas, we wanted to have a structured application. First, we thought of starting with nx, but we chose [turborepo](https://github.com/Montekkundan/hackmath) to hold all our projects (blog, hackmath, admin).

Hackmath website is built using Nextjs 14 app router, tailwind, and uses drizzle orm to connect with Neon DB. It uses Vercel AI SDK to use Google Gemini API for chat as well as generative AI.

Dr Ham, the AI that helps students with Math, is built in 2 ways, the basic version and the premium version. Both versions use the Gemini Pro model, but the premium one has generative UI capabilities.

We added Stripe to manage switching between the primary and premium versions.

The authentication is built using next-auth. We started with clerk authentication but later created our own custom system because we need roles (user, teacher, parent, organization) based on future plans.
The system offers two-factor authentication and settings to change passwords; a forgot password system is implemented, too.
We are using Resent API to send emails to users for all authentication-related functions.

We are using [Neon db](https://neon.tech) for our free database. Drizzle ORM for connection.

Our [blog](https://blog.hackmath.org) is created using [Astro](https://astro.build). We first had our blog on our hackmath website using Notion Api, but later decided to shift to a proper website. Why Astro? Because it's best for creating blog-like websites.

All the color schemes and web design can be viewed in our public [Figma file](https://www.figma.com/file/s8L3Tktpd5ImhwqTVN2FYv/Gemini-Hack--HackMath?type=design&node-id=0-1&mode=design&t=kyO3kwzKKWgHdAKX-0)

# Challenges we ran into

The first challenge was which database to use as Planet-scale removed their free tier. We found out Neon db, and instead of using Prisma, we are using Drizzle as we found it better.

Using Turborepo for the entire project was challenging because it used quite a few resources, and building the project every time took a lot of time. Still, Turborepo is a good choice as we will have many of the same components we will add in the future, and it has built caching.

Vercel AI SDK for generative UI was problematic as the Gemini vertical example uses the 3.0.17 version of the AI package to demonstrate generative UI. We are getting the ` Package path ./google is not exported from package` error, which is [documented in the package repo](https://github.com/vercel/ai/issues/1450)

Building the blog was easy, but as we use PNPM, there were build issues initially with Vercel. We fixed it! 🚀

# Accomplishments that we're proud of

We've successfully harnessed the power of AI to make Dr. Ham. We are proud to use Turborepo, which helps us see all projects in one place.

We developed our own custom authentication system using resend to use email authentication. We also added roles that will be used in future updates for HackMath.

We have created this application through deep research and UI development. In collaborative works, we are proud of completing the initial MVP development.

Lastly, we are happy to use Gemini, and it powers Dr. Ham!

# Business Potential

> [Business Pitch Deck](https://drive.google.com/file/d/1aWUdf6w8sne3w4s8Ws_l3JN1VHL95s8Y/view?usp=sharing)

Here is our pitch deck on how _HackMath_ plans to grow as a business.

> _"Current Global STEM education market is **$37.8 billion** in K-12. STEM tutoring market including math is expected to reach **$86.7 billion** by 2028(Statistia, 2021)."_

By assisting teachers in helping students keep on track with math, we expect to help _50.2 million_ US students from 1st-12th grade receive 313.248[2] billion worth of private math education annually for 1/50th price.

![Pitchdeck](https://i.imgur.com/Z8TvZi5.png)

# What's next for HackMath

**HackMath** is developing administrative features for teachers, allowing them to customize the HackMath path for students based on their course structures.

We plan to conduct market research and user testing by partnering with high schools in Arizona, United States, and British Columbia, Canada. After customizing the system based on user feedback and advisory input from math education professors, we will release our service by December 2024.

**Dev Note**
We plan to fix the generative UI challenges that we are facing and add/update user data collection to have AI-generated analytics.

Follow our Instagram for more updates!
@hackmath.official

#Appendix
[1] US math teacher-to-student ratios
1,413,345(US Math Teacher)(Zippia, n.d) : 50.2M(US Students) = 1 : 35.5

[2]: $313.248B = 50.2M students(US students) x $40(Average Private Math Tutoring Price) x 3 times(Most effective frequency of tutoring per week) x 52 weeks(Total Weeks per Year)

# References

Research Gate. (2022). The effect of immediate feedback on mathematics learning achievement. Retrieved from https://www.researchgate.net/publication/365875713_The_effect_of_immediate_feedback_on_mathematics_learning_achievement

Wirth, A. S. (2022, October 24). Math and reading scores plummet as the pandemic disrupts education. The New York Times. Retrieved from https://www.nytimes.com/2022/10/24/us/math-reading-scores-pandemic.html

US Census Bureau. (2023). ACS 55: An analysis of educational attainment in the United States. Retrieved from https://www.census.gov/content/dam/Census/library/publications/2023/acs/acs-55.pdf

Harvard Graduate School of Education. (2023, May 23). New data show how the pandemic affected learning across whole communities. Retrieved from https://www.gse.harvard.edu/ideas/news/23/05/new-data-show-how-pandemic-affected-learning-across-whole-communities

Tutors.com. (n.d.). Math tutor prices. Retrieved from https://tutors.com/costs/math-tutor-prices

U.S. Department of Education, National Center for Education Statistics (NCES). (n.d.). Table 208.40. Public elementary and secondary teachers: Student-to-teacher ratio, by subject area and grade level: Selected years, 1990-91 through 2020-21: https://nces.ed.gov/programs/digest/d20/tables/dt20_208.40.asp

Zippia. (n.d.). Math Teacher Jobs: Demographics: https://www.zippia.com/math-teacher-jobs/demographics/
