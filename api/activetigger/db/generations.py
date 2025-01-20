import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import sessionmaker

from activetigger.datamodels import GenerationModel
from activetigger.db.models import Generations


class GenerationsService:
    Session: sessionmaker[SessionType]

    def __init__(self, sessionmaker: sessionmaker[SessionType]):
        self.Session = sessionmaker

    def add_generated(
        self,
        user: str,
        project_slug: str,
        element_id: str,
        endpoint: str,
        prompt: str,
        answer: str,
    ):
        session = self.Session()
        generation = Generations(
            user_id=user,
            time=datetime.datetime.now(),
            project_id=project_slug,
            element_id=element_id,
            endpoint=endpoint,
            prompt=prompt,
            answer=answer,
        )
        session.add(generation)
        session.commit()
        session.close()

    def get_generated(self, project_slug: str, username: str, n_elements: int = 10):
        """
        Get elements from generated table by order desc
        """
        with self.Session() as session:
            generated = session.scalars(
                select(Generations)
                .filter_by(project_id=project_slug, user_id=username)
                .order_by(Generations.time.desc())
                .limit(n_elements)
            ).all()
            print(generated)
            return [[el.time, el.element_id, el.prompt, el.answer, el.endpoint] for el in generated]

    def get_available_models(self):
        """
        Get the available models for generation

        Currently, this is hardwired in code
        """
        return [
            GenerationModel(id="ollama", name="Ollama"),
            GenerationModel(id="gpt-4o-mini", name="ChatGPT 4o mini"),
        ]
