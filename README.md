<!-- Improved compatibility of back to top link: See: https://github.com/dhmnr/skipr/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">skipr</h3>

  <p align="center">
    A novel decoding method for LLM inference that skips the non-essenial layers using a learned RL policy.
    <br />
    <a href="https://github.com/dhmnr/skipr"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/dhmnr/skipr">View Demo</a>
    ·
    <a href="https://github.com/dhmnr/skipr/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/dhmnr/skipr/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
skipr is an advanced method designed to optimize the performance of large language models (LLMs) by dynamically skipping layers based on a reinforcement learning (RL) policy with a reward model. This approach improves inference speed and computational efficiency while maintaining model accuracy.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Features
 - Dynamic Layer Skipping: Utilizes an RL policy to decide which layers to skip during inference, reducing computational overhead.
 - Reward-Based Optimization: Employs a reward model to evaluate the effectiveness of skipped layers, ensuring that performance is not compromised.
 - Adaptive Inference: Continuously learns and adapts to different tasks and contexts, optimizing layer usage for each specific scenario.
 - Easy Integration: Designed to integrate seamlessly with existing LLM frameworks and architectures.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### How It Works
The skipr initializes with a pre-trained LLM and a reward model. The reward model evaluates the impact of skipping layers on the overall performance.

An RL policy is used to determine which layers to skip based on the reward model's feedback. We use a policy gradient method to maximize the reward, the reward function being $$R = - L + \alpha \cdot t$$ 
where,

$L$ is model loss  
$t$ is the number if skipped layers  
$\alpha$ is a hyperparameter aimed to balance minmizing loss while maximizing skipped layers.

During inference, the tool dynamically skips layers as determined by the RL policy, allowing for faster inference. The reward model provides feedback on the performance of skipped layers, enabling the RL policy to adjust its decisions and improve over time.



<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- GETTING STARTED -->
## Installation and Usage


<!-- ### Usage -->

This project use poetry for dependency management, and is required for running skipr locally. 

1. Clone the repo
   ```sh
   git clone https://github.com/dhmnr/skipr.git
   ```
2. Install Dependencies
   ```sh
   poetry install
   ```

4. Train the classifier for BERT and later tain the policy network or run the script `run.sh`
   ```sh
    export TASK_NAME=rte
    export EPOCHS=1000

    poetry run glue \
    --model_name_or_path google-bert/bert-base-cased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs $EPOCHS \
    --output_dir ./models/$TASK_NAME/classifier \
    --overwrite_output_dir \
    --mode classifier


    poetry run glue \
    --model_name_or_path ./models/$TASK_NAME/classifier \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-4 \
    --num_train_epochs $EPOCHS \
    --output_dir ./models/$TASK_NAME/policy \
    --overwrite_output_dir \
    --mode policy \
    --alpha 0.07
   ``` 


<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- ROADMAP -->
<!-- ## Roadmap



See the [open issues](https://github.com/dhmnr/skipr/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<a href="https://github.com/dhmnr/skipr/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dhmnr/skipr" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GNU GPLv3 License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

<!-- Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com -->

Project Link: [https://github.com/dhmnr/skipr](https://github.com/dhmnr/skipr)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/dhmnr/skipr.svg?style=for-the-badge
[contributors-url]: https://github.com/dhmnr/skipr/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dhmnr/skipr.svg?style=for-the-badge
[forks-url]: https://github.com/dhmnr/skipr/network/members
[stars-shield]: https://img.shields.io/github/stars/dhmnr/skipr.svg?style=for-the-badge
[stars-url]: https://github.com/dhmnr/skipr/stargazers
[issues-shield]: https://img.shields.io/github/issues/dhmnr/skipr.svg?style=for-the-badge
[issues-url]: https://github.com/dhmnr/skipr/issues
[license-shield]: https://img.shields.io/github/license/dhmnr/skipr.svg?style=for-the-badge
[license-url]: https://github.com/dhmnr/skipr/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
<!-- [Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/ -->

