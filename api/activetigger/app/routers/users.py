from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from activetigger.app.dependencies import oauth2_scheme, test_rights, verified_user
from activetigger.datamodels import (
    AuthActions,
    UserInDBModel,
    UserModel,
    UsersServerModel,
    UserStatistics,
)
from activetigger.orchestrator import orchestrator

router = APIRouter()


@router.post("/users/disconnect", dependencies=[Depends(verified_user)], tags=["users"])
async def disconnect_user(token: Annotated[str, Depends(oauth2_scheme)]) -> None:
    """
    Revoke user connexion
    """
    try:
        orchestrator.revoke_access_token(token)
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/me", tags=["users"])
async def read_users_me(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> UserModel:
    """
    Information on current user
    """
    try:
        return UserModel(username=current_user.username, status=current_user.status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users", tags=["users"])
async def existing_users(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> UsersServerModel:
    """
    Get existing users
    """
    try:
        users = orchestrator.users.existing_users()
        return UsersServerModel(
            users=users,
            auth=["manager", "annotator"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/recent", tags=["users"])
async def recent_users() -> list[str]:
    """
    Get recently connected users
    """
    return orchestrator.db_manager.users_service.get_current_users(300)


@router.post("/users/create", dependencies=[Depends(verified_user)], tags=["users"])
async def create_user(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    username_to_create: str = Query(),
    password: str = Query(),
    status: str = Query(),
    mail: str = Query(),
) -> None:
    """
    Create user
    """
    test_rights("create user", current_user.username)
    try:
        orchestrator.users.add_user(
            username_to_create, password, status, current_user.username, mail
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return None


@router.post("/users/delete", dependencies=[Depends(verified_user)], tags=["users"])
async def delete_user(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], user_to_delete: str
) -> None:
    """
    Delete user
    - root can delete all
    - users can only delete account they created
    """
    test_rights("modify user", current_user.username)
    try:
        orchestrator.users.delete_user(user_to_delete, current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return None


@router.post("/users/changepwd", dependencies=[Depends(verified_user)], tags=["users"])
async def change_password(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    pwdold: str = Query(),
    pwd1: str = Query(),
    pwd2: str = Query(),
) -> None:
    """
    Change password for an account
    """
    try:
        orchestrator.users.change_password(current_user.username, pwdold, pwd1, pwd2)
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/users/auth/{action}", dependencies=[Depends(verified_user)], tags=["users"]
)
async def set_auth(
    action: AuthActions,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    username: str = Query(),
    project_slug: str = Query(),
    status: str = Query(None),
) -> None:
    """
    Modify user auth on a specific project
    """
    test_rights("modify project", current_user.username, project_slug)
    if action == "add":
        if not status:
            raise HTTPException(status_code=400, detail="Missing status")
        try:
            orchestrator.users.set_auth(username, project_slug, status)
            orchestrator.log_action(
                current_user.username, f"INFO add user {username}", "all"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        return None

    if action == "delete":
        try:
            orchestrator.users.delete_auth(username, project_slug)
            orchestrator.log_action(
                current_user.username, f"INFO delete user {username}", "all"
            )
        except Exception as e:
            raise HTTPException(status_code=500) from e

        return None

    raise HTTPException(status_code=400, detail="Action not found")


@router.get("/users/auth", dependencies=[Depends(verified_user)], tags=["users"])
async def get_auth(username: str) -> list:
    """
    Get all user auth
    """
    try:
        return orchestrator.users.get_auth(username, "all")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/users/statistics", dependencies=[Depends(verified_user)], tags=["users"])
async def get_statistics(username: str) -> UserStatistics:
    """
    Get statistics for specific user
    """
    try:
        return orchestrator.users.get_statistics(username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
