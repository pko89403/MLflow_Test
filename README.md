# MLFlow
## MLproject로 전체 프로젝트 컨트롤이 가능하다.
## 기능은 Logging + Project Control
### 도커 빌드 하면 그럼 도커가 일을 함.
**학습 실행 CMD**    
~~~sh
mlflow run . -e train
~~~
**Nginx-Gunicorn-Flask API Endpoint 실행 CMD**   
~~~sh
mlflow run . -e serve
~~~
### 뒤에 이어서 no-conda 버전이랑 다양하게 진행한 경험이 있음.
