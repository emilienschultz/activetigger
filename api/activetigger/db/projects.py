import datetime
import logging
from typing import Any, TypedDict

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session as SessionType
from sqlalchemy.orm import sessionmaker

from activetigger.datamodels import FeatureDescriptionModel
from activetigger.db import DBException
from activetigger.db.models import (
    Annotations,
    Auths,
    Features,
    Projects,
    Schemes,
    Tokens,
)


class Codebook(TypedDict):
    codebook: str
    time: str


class ProjectsService:
    Session: sessionmaker[SessionType]

    def __init__(self, sessionmaker: sessionmaker[SessionType]):
        self.Session = sessionmaker

    def get_project(self, project_slug: str):
        session = self.Session()
        project = session.query(Projects).filter_by(project_slug=project_slug).first()
        session.close()
        if project:
            return project.__dict__
        else:
            return None

    def add_project(
        self, project_slug: str, parameters: dict[str, Any], username: str
    ) -> str:
        with self.Session.begin() as session:
            now = datetime.datetime.now()
            project = Projects(
                project_slug=project_slug,
                parameters=parameters,
                time_created=now,
                time_modified=now,
                user_id=username,
            )
            session.add(project)
        logging.debug("CREATE PROJECT %s", now)
        return project_slug

    def update_project(self, project_slug: str, parameters: dict[str, Any]):
        with self.Session.begin() as session:
            project = (
                session.query(Projects).filter_by(project_slug=project_slug).first()
            )
            if project is None:
                raise DBException("Project not found")

            project.time_modified = datetime.datetime.now()
            project.parameters = parameters
            session.commit()

    def existing_projects(self) -> list[str]:
        session = self.Session()
        projects = session.query(Projects).all()
        session.close()
        return [project.project_slug for project in projects]

    def add_token(self, token: str, status: str):
        with self.Session.begin() as session:
            new_token = Tokens(
                token=token, status=status, time_created=datetime.datetime.now()
            )
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

    def add_scheme(
        self, project_slug: str, name: str, labels: list[str], kind: str, username: str
    ):
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

    def update_scheme_codebook(
        self, project_slug: str, scheme: str, codebook: str
    ) -> None:
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
            project = session.scalars(
                select(Projects).filter_by(project_slug=project_slug)
            ).first()
            session.delete(project)

    def get_project_auth(self, project_slug: str):
        with self.Session() as session:
            auth = session.scalars(
                select(Auths).filter_by(project_id=project_slug)
            ).all()
            return {el.user_id: el.status for el in auth}

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
            _ = session.execute(
                delete(Auths).filter_by(project_id=project_slug, user_id=user)
            )

    def get_user_auth_projects(self, username: str):
        """
        Projects user can access (auth)
        """
        with self.Session() as session:
            result = session.execute(
                select(
                    Auths.project_id,
                    Auths.status,
                    Projects.parameters,
                    Projects.user_id,
                    Projects.time_created,
                )
                .join(Auths.project)
                .where(Auths.user_id == username)
            ).all()
            return result

    def get_user_auth(self, username: str, project_slug: str | None = None):
        session = self.Session()
        if project_slug is None:
            result = (
                session.query(Auths.user_id, Auths.status)
                .filter(Auths.user_id == username)
                .all()
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

    def get_recent_annotations(
        self, project_slug: str, user: str, scheme: str, limit: int
    ):
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
        elements: list[
            dict
        ],  # [{"element_id": str, "annotation": str, "comment": str}]
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
        annotation: str | None,
        comment: str | None = "",
        selection: str | None = None,
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
                selection=selection,
            )
            session.add(new_annotation)

    def available_schemes(self, project_slug: str):
        with self.Session() as session:
            schemes = session.execute(
                select(Schemes.name, Schemes.params)
                .filter_by(project_id=project_slug)
                .distinct()
            ).all()
        r = []
        for s in schemes:
            params = s.params
            kind = (
                params["kind"] if "kind" in params else "multiclass"
            )  # temporary hack
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
            _ = session.execute(
                delete(Schemes).filter_by(name=name, project_id=project_slug)
            )

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
            return [
                [row.element_id, row.annotation, row.user_id, row.time]
                for row in results
            ]

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
        with self.Session.begin() as session:
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

    def delete_feature(self, project: str, name: str):
        session = self.Session()
        session.query(Features).filter(
            Features.name == name, Features.project_id == project
        ).delete()
        session.commit()
        session.close()

    def delete_all_features(self, project: str):
        with self.Session.begin() as session:
            session.query(Features).filter(Features.project_id == project).delete()

    def get_feature(self, project: str, name: str):
        session = self.Session()
        feature = (
            session.query(Features)
            .filter(Features.name == name, Features.project_id == project)
            .first()
        )
        session.close()
        return feature

    def get_project_features(self, project: str) -> dict[str, FeatureDescriptionModel]:
        with self.Session() as session:
            features = session.scalars(
                select(Features).filter_by(project_id=project)
            ).all()
            return {
                i.name: FeatureDescriptionModel(
                    time=i.time.strftime("%Y-%m-%d %H:%M:%S"),
                    kind=i.kind,
                    parameters=i.parameters,
                    user=str(i.user_id),  # check
                    cols=list(i.data),
                    name=i.name,
                )
                for i in features
            }
