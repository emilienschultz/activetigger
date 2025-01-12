import datetime
import logging
from collections.abc import Sequence
from typing import Any, TypedDict

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import sessionmaker

from activetigger.db import DBException
from activetigger.db.models import (
    Annotations,
    Auths,
    Features,
    Generations,
    Logs,
    Models,
    Projects,
    Schemes,
    Tokens,
    Users,
)


class Codebook(TypedDict):
    codebook: str
    time: str


class ProjectsService:
    Session: sessionmaker[SessionType]

    def __init__(self, sessionmaker: sessionmaker[SessionType]):
        self.Session = sessionmaker

    def add_log(self, user: str, action: str, project_slug: str, connect: str):
        session = self.Session()
        log = Logs(
            user_id=user,
            project_id=project_slug,
            action=action,
            connect=connect,
            time=datetime.datetime.now(),
        )
        session.add(log)
        session.commit()
        session.close()

    def get_logs(self, username: str, project_slug: str, limit: int):
        with self.Session() as session:
            stmt = select(Logs).order_by(Logs.time.desc()).limit(limit)
        if project_slug != "all":
            stmt = stmt.filter_by(project_id=project_slug)
        if username != "all":
            stmt = stmt.filter_by(user_id=username)

        logs = session.scalars(stmt).all()

        return [
            {
                "id": log.id,
                "time": log.time,
                "user": log.user_id,
                "project": log.project_id,
                "action": log.action,
                "connect": log.connect,
            }
            for log in logs
        ]

    def get_project(self, project_slug: str):
        session = self.Session()
        project = session.query(Projects).filter_by(project_slug=project_slug).first()
        session.close()
        if project:
            return project.__dict__
        else:
            return None

    def add_project(self, project_slug: str, parameters: dict[str, Any], username: str):
        session = self.Session()
        project = Projects(
            project_slug=project_slug,
            parameters=parameters,
            time_created=datetime.datetime.now(),
            time_modified=datetime.datetime.now(),
            user_id=username,
        )
        session.add(project)
        session.commit()
        session.close()
        print("CREATE PROJECT", datetime.datetime.now())

    def update_project(self, project_slug: str, parameters: dict[str, Any]):
        session = self.Session()
        project = session.query(Projects).filter_by(project_slug=project_slug).first()
        if project is None:
            raise DBException("Project not found")

        project.time_modified = datetime.datetime.now()
        project.parameters = parameters
        session.commit()
        session.close()

    def existing_projects(self) -> list[str]:
        session = self.Session()
        projects = session.query(Projects).all()
        session.close()
        return [project.project_slug for project in projects]

    def add_token(self, token: str, status: str):
        with self.Session.begin() as session:
            new_token = Tokens(token=token, status=status, time_created=datetime.datetime.now())
            session.add(new_token)

    def get_token_status(self, token: str):
        with self.Session() as session:
            found_token = session.scalars(select(Tokens).filter_by(token=token)).first()
            if found_token is None:
                raise DBException("Token not found")
            return found_token.status

    def revoke_token(self, token: str):
        with self.Session.begin() as session:
            _ = session.execute(
                update(Tokens)
                .filter_by(token=token)
                .values(time_revoked=datetime.datetime.now(), status="revoked")
            )

    def add_scheme(self, project_slug: str, name: str, labels: list[str], kind: str, username: str):
        if not labels:
            labels = []
        params = {"labels": labels, "codebook": None, "kind": kind}
        with self.Session.begin() as session:
            scheme = Schemes(
                project_id=project_slug,
                name=name,
                params=params,
                user_id=username,
                time_created=datetime.datetime.now(),
                time_modified=datetime.datetime.now(),
            )
            session.add(scheme)

    def update_scheme_labels(self, project_slug: str, name: str, labels: list[str]):
        """
        Update the labels in the database
        """
        with self.Session.begin() as session:
            scheme = session.scalars(
                select(Schemes).filter_by(project_id=project_slug, name=name)
            ).first()
            if scheme is None:
                raise DBException("Scheme not found")
            params = scheme.params.copy()
            params["labels"] = labels
            scheme.params = params
            scheme.time_modified = datetime.datetime.now()

    def update_scheme_codebook(self, project_slug: str, scheme: str, codebook: str) -> None:
        """
        Update the codebook in the database
        """
        logging.debug(f"update_scheme_codebook {project_slug} {scheme}")
        with self.Session.begin() as session:
            result_scheme = session.scalars(
                select(Schemes).filter_by(project_id=project_slug, name=scheme)
            ).first()
            if result_scheme is None:
                raise DBException("Scheme not found")
            params = result_scheme.params.copy()
            params["codebook"] = codebook
            result_scheme.params = params
            result_scheme.time_modified = datetime.datetime.now()

    def get_scheme_codebook(self, project_slug: str, name: str) -> Codebook:
        with self.Session() as session:
            scheme = session.scalars(
                select(Schemes).filter_by(project_id=project_slug, name=name)
            ).first()
            if scheme is None:
                raise DBException("Scheme not found")
            return {
                "codebook": scheme.params["codebook"],
                "time": str(scheme.time_modified),
            }

    def delete_project(self, project_slug: str):
        with self.Session.begin() as session:
            _ = session.execute(delete(Projects).filter_by(project_id=project_slug))

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
        session = self.Session()
        generated = (
            session.query(Generations)
            .filter(Generations.project_id == project_slug, Generations.user_id == username)
            .order_by(Generations.time.desc())
            .limit(n_elements)
            .all()
        )
        session.close()
        return [[el.time, el.element_id, el.prompt, el.answer, el.endpoint] for el in generated]

    def get_distinct_users(self, project_slug: str, timespan: int | None) -> Sequence[Users]:
        with self.Session() as session:
            stmt = (
                select(Projects.user)
                .join_from(Projects, Users)
                .where(Projects.project_slug == project_slug)
                .distinct()
            )
            if timespan:
                time_threshold = datetime.datetime.now() - datetime.timedelta(seconds=timespan)
                stmt = stmt.join(Annotations).where(
                    Annotations.time > time_threshold,
                )
        return session.scalars(stmt).all()

    def get_current_users(self, timespan: int = 600):
        session = self.Session()
        time_threshold = datetime.datetime.now() - datetime.timedelta(seconds=timespan)
        users = session.query(Logs.user).filter(Logs.time > time_threshold).distinct().all()
        session.close()
        return [u.user for u in users]

    def get_project_auth(self, project_slug: str):
        with self.Session() as session:
            auth = session.scalars(select(Auths).filter_by(project_id=project_slug)).all()
            return {el.user: el.status for el in auth}

    def add_auth(self, project_slug: str, user: str, status: str):
        with self.Session.begin() as session:
            auth = session.scalars(
                select(Auths).filter_by(project_id=project_slug, user_id=user)
            ).first()
            if auth is not None:
                auth.status = status
            else:
                auth = Auths(project_id=project_slug, user_id=user, status=status)
                session.add(auth)

    def delete_auth(self, project_slug: str, user: str):
        with self.Session.begin() as session:
            _ = session.execute(delete(Auths).filter_by(project_id=project_slug, user_id=user))

    def get_user_projects(self, username: str):
        with self.Session() as session:
            result = session.execute(
                select(
                    Auths.project,
                    Auths.status,
                    Projects.parameters,
                    Projects.user_id,
                    Projects.time_created,
                )
                .join(Auths.project)
                .where(Auths.user_id == username)
            ).all()
            return [row for row in result]

    def get_user_auth(self, username: str, project_slug: str | None = None):
        session = self.Session()
        if project_slug is None:
            result = (
                session.query(Auths.user_id, Auths.status).filter(Auths.user_id == username).all()
            )
        else:
            result = (
                session.query(Auths.user_id, Auths.status)
                .filter(Auths.user_id == username, Auths.project_id == project_slug)
                .all()
            )
        session.close()
        return [[row[0], row[1]] for row in result]

    def get_scheme_elements(self, project_slug: str, scheme: str, dataset: list[str]):
        """
        Get last annotation for each element id for a project/scheme
        """
        with self.Session() as session:
            results = session.execute(
                select(
                    Annotations.element_id,
                    Annotations.annotation,
                    Annotations.user_id,
                    Annotations.time,
                    Annotations.comment,
                    func.max(Annotations.time),
                )
                .filter_by(scheme_id=scheme, project_id=project_slug)
                .where(Annotations.dataset.in_(dataset))
                .group_by(Annotations.element_id)
                .order_by(func.max(Annotations.time).desc())
            )

            # Execute the query and fetch all results
            return [
                [row.element_id, row.annotation, row.user_id, row.time, row.comment]
                for row in results
            ]

    def get_coding_users(self, scheme: str, project_slug: str) -> Sequence[Users]:
        with self.Session() as session:
            distinct_users = session.scalars(
                select(Annotations.user)
                .join_from(Annotations, Users)
                .where(
                    Annotations.project_id == project_slug,
                    Annotations.scheme_id == scheme,
                )
                .distinct()
            ).all()
            return distinct_users

    def get_recent_annotations(self, project_slug: str, user: str, scheme: str, limit: int):
        with self.Session() as session:
            stmt = (
                select(Annotations.element_id)
                .filter_by(
                    project_id=project_slug,
                    scheme_id=scheme,
                    dataset="train",
                )
                .order_by(Annotations.time.desc())
                .limit(limit)
                .distinct()
            )
            if user != "all":
                stmt = stmt.filter_by(user_id=user)
            recent_annotations = session.execute(stmt)
            return [u[0] for u in recent_annotations]

    def get_annotations_by_element(
        self, project_slug: str, scheme: str, element_id: str, limit: int = 10
    ):
        with self.Session() as session:
            annotations = session.execute(
                select(
                    Annotations.annotation,
                    Annotations.dataset,
                    Annotations.user_id,
                    Annotations.time,
                )
                .filter_by(
                    project_id=project_slug,
                    scheme_id=scheme,
                    element_id=element_id,
                )
                .order_by(Annotations.time.desc())
                .limit(limit)
            ).all()
            return [[a.annotation, a.dataset, a.user_id, a.time] for a in annotations]

    def add_annotations(
        self,
        dataset: str,
        user: str,
        project_slug: str,
        scheme: str,
        elements: list[dict],  # [{"element_id": str, "annotation": str, "comment": str}]
    ):
        session = self.Session()
        for e in elements:
            annotation = Annotations(
                time=datetime.datetime.now(),
                dataset=dataset,
                user_id=user,
                project_id=project_slug,
                element_id=e["element_id"],
                scheme_id=scheme,
                annotation=e["annotation"],
                comment=e["comment"],
            )
            session.add(annotation)
        session.commit()
        session.close()

    def add_annotation(
        self,
        dataset: str,
        user: str,
        project_slug: str,
        element_id: str,
        scheme: str,
        annotation: str,
        comment: str = "",
    ):
        with self.Session.begin() as session:
            new_annotation = Annotations(
                time=datetime.datetime.now(),
                dataset=dataset,
                user_id=user,
                project_id=project_slug,
                element_id=element_id,
                scheme_id=scheme,
                annotation=annotation,
                comment=comment,
            )
            session.add(new_annotation)

    def available_schemes(self, project_slug: str):
        with self.Session() as session:
            schemes = session.execute(
                select(Schemes.name, Schemes.params).filter_by(project_id=project_slug).distinct()
            ).all()
        r = []
        for s in schemes:
            params = s.params
            kind = params["kind"] if "kind" in params else "multiclass"  # temporary hack
            r.append(
                {
                    "name": s.name,
                    "labels": params["labels"],
                    "codebook": params["codebook"],
                    "kind": kind,
                }
            )
        return r

    def delete_scheme(self, project_slug: str, name: str):
        with self.Session.begin() as session:
            _ = session.execute(delete(Schemes).filter_by(name=name, project_id=project_slug))

    def get_table_annotations_users(self, project_slug: str, scheme: str):
        with self.Session() as session:
            subquery = (
                select(
                    Annotations.id,
                    Annotations.user_id,
                    func.max(Annotations.time).label("last_timestamp"),
                )
                .filter_by(project_id=project_slug, scheme_id=scheme)
                .group_by(Annotations.element_id, Annotations.user_id)
                .subquery()
            )
            query = select(
                Annotations.element_id,
                Annotations.annotation,
                Annotations.user_id,
                Annotations.time,
            ).join(subquery, Annotations.id == subquery.c.id)

            results = session.execute(query).fetchall()
            return [[row.element_id, row.annotation, row.user_id, row.time] for row in results]

    # feature management

    def add_feature(
        self,
        project: str,
        kind: str,
        name: str,
        parameters: dict[str, Any],
        user: str,
        data: list[dict[str, Any]] | None = None,
    ):
        session = self.Session()
        feature = Features(
            project_id=project,
            time=datetime.datetime.now(),
            kind=kind,
            name=name,
            parameters=parameters,
            user_id=user,
            data=data,
        )
        session.add(feature)
        session.commit()
        session.close()

    def delete_feature(self, project: str, name: str):
        session = self.Session()
        session.query(Features).filter(
            Features.name == name, Features.project_id == project
        ).delete()
        session.commit()
        session.close()

    def get_feature(self, project: str, name: str):
        session = self.Session()
        feature = (
            session.query(Features)
            .filter(Features.name == name, Features.project_id == project)
            .first()
        )
        session.close()
        return feature

    def get_project_features(self, project: str):
        with self.Session() as session:
            features = session.scalars(select(Features).filter_by(project_id=project)).all()
            return {
                i.name: {
                    "time": i.time.strftime("%Y-%m-%d %H:%M:%S"),
                    "kind": i.kind,
                    "parameters": i.parameters,
                    "user": i.user,
                    "data": i.data,
                }
                for i in features
            }

    def add_model(
        self,
        kind: str,
        project: str,
        name: str,
        user: str,
        status: str,
        scheme: str,
        params: dict[str, Any],
        path: str,
    ):
        session = self.Session()

        # test if the name does not exist
        models = session.query(Models).filter(Models.name == name).all()
        if len(models) > 0:
            return False

        model = Models(
            project_id=project,
            time=datetime.datetime.now(),
            kind=kind,
            name=name,
            user_id=user,
            parameters=params,
            scheme_id=scheme,
            status=status,
            path=path,
        )
        session.add(model)
        session.commit()
        session.close()

        print("available", self.available_models(project))

        return True

    def change_model_status(self, project: str, name: str, status: str):
        with self.Session.begin() as session:
            _ = session.execute(
                update(Models).filter_by(name=name, project_id=project).values(status=status)
            )

    def available_models(self, project: str):
        with self.Session() as session:
            models = session.execute(
                select(Models.name, Models.parameters, Models.path, Models.scheme_id)
                .filter_by(
                    project_id=project,
                    status="trained",
                )
                .distinct()
            ).all()
        return [
            {
                "name": m.name,
                "scheme": m.scheme,
                "path": m.path,
                "parameters": m.parameters,
            }
            for m in models
        ]

    def model_exists(self, project: str, name: str):
        session = self.Session()
        models = (
            session.query(Models).filter(Models.name == name, Models.project_id == project).all()
        )
        session.close()
        return len(models) > 0

    def delete_model(self, project: str, name: str):
        session = self.Session()
        # test if the name does not exist
        models = (
            session.query(Models).filter(Models.name == name, Models.project_id == project).all()
        )
        if len(models) == 0:
            print("Model does not exist")
            return False
        # delete the model
        session.query(Models).filter(Models.name == name, Models.project_id == project).delete()
        session.commit()
        session.close()
        return True

    def get_model(self, project: str, name: str):
        session = self.Session()
        model = (
            session.query(Models).filter(Models.name == name, Models.project_id == project).first()
        )
        session.close()
        return model

    def rename_model(self, project: str, old_name: str, new_name: str):
        session = self.Session()

        # test if the name does not exist
        models = (
            session.query(Models)
            .filter(Models.name == new_name, Models.project_id == project)
            .all()
        )
        if len(models) > 0:
            return {"error": "The new name already exists"}
        # get and rename
        model = (
            session.query(Models)
            .filter(Models.name == old_name, Models.project_id == project)
            .first()
        )
        if model is None:
            raise DBException("Model not found")

        model.name = new_name
        model.path = model.path.replace(old_name, new_name)
        session.commit()
        session.close()
        return {"success": "model renamed"}

    def set_model_params(self, project: str, name: str, flag: str, value):
        session = self.Session()
        model = (
            session.query(Models).filter(Models.name == name, Models.project_id == project).first()
        )
        if model is None:
            raise DBException("Model not found")

        parameters = model.parameters
        parameters[flag] = value
        session.commit()
