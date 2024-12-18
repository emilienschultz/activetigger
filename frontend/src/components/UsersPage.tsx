import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { MdOutlineDeleteOutline } from 'react-icons/md';

import Select from 'react-select';
import {
  useAddUserAuthProject,
  useCreateUser,
  useDeleteUser,
  useDeleteUserAuthProject,
  useUserProjects,
  useUsers,
  useUsersAuth,
} from '../core/api';
import { useNotifications } from '../core/notifications';
import { ChangePassword } from './forms/ChangePassword';
import { PageLayout } from './layout/PageLayout';

interface newUser {
  username: string;
  password: string;
  status: string;
  mail: string;
}

export const UsersPage: FC = () => {
  const projects = useUserProjects();
  const [currentProjectSlug, setCurrentProjectSlug] = useState<string | null>(null);
  const { notify } = useNotifications();

  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [currentAuth, setCurrentAuth] = useState<string | null>(null);

  const { authUsers, reFetchUsersAuth } = useUsersAuth(currentProjectSlug);
  const { users, reFetchUsers } = useUsers();

  const { deleteUser } = useDeleteUser(reFetchUsers);
  const { createUser } = useCreateUser(reFetchUsers);

  const { deleteUserAuth } = useDeleteUserAuthProject(currentProjectSlug, reFetchUsersAuth);
  const { addUserAuth } = useAddUserAuthProject(currentProjectSlug, reFetchUsersAuth);

  const { handleSubmit, register, reset } = useForm<newUser>();
  const onSubmit: SubmitHandler<newUser> = async (data) => {
    await createUser(data.username, data.password, data.status, data.mail);
    reset();
  };

  const userOptions = users
    ? Object.keys(users).map((userKey) => ({
        value: userKey,
        label: userKey,
      }))
    : [];

  console.log(users);

  return (
    <PageLayout currentPage="users">
      <div className="container-fluid">
        <div className="row">
          <div className="col-1"></div>
          <div className="col-8">
            <ChangePassword />
            <h2 className="subsection">Manage users and rights</h2>

            <div className="explanations">
              Select a user (you can only delete users you created)
            </div>

            <div className="d-flex align-items-center">
              <Select
                id="select-user"
                className="form-select"
                options={userOptions}
                onChange={(selectedOption) => {
                  setCurrentUser(selectedOption ? selectedOption.value : null);
                }}
                isClearable
                placeholder="Select a user"
              />
              <button
                className="btn btn p-0"
                onClick={() => {
                  deleteUser(currentUser);
                  reFetchUsers();
                }}
              >
                <MdOutlineDeleteOutline size={30} />
              </button>
            </div>
            <details className="custom-details">
              <summary>Add user</summary>
              <form onSubmit={handleSubmit(onSubmit)}>
                <input
                  className="form-control me-2 mt-2"
                  type="text"
                  {...register('username')}
                  placeholder="New user name"
                />
                <input
                  className="form-control me-2 mt-2"
                  type="text"
                  {...register('password')}
                  placeholder="Password"
                />
                <input
                  className="form-control me-2 mt-2"
                  type="email"
                  {...register('mail')}
                  placeholder="Mail"
                />
                <select {...register('status')} className="me-2 mt-2">
                  <option>manager</option>
                  <option>annotator</option>
                </select>
                <button className="btn btn-primary me-2 mt-2">Add user</button>
              </form>
            </details>
            <span className="explanations">Select the project</span>

            <br></br>
            {
              <select
                className="form-select"
                onChange={(e) => {
                  setCurrentProjectSlug(e.target.value);
                }}
              >
                <option></option>
                {(projects || []).map((project) => (
                  <option key={project.parameters.project_slug}>
                    {project.parameters.project_slug}
                  </option>
                ))}
              </select>
            }
            <div>
              {authUsers ? (
                <table className="table-auth">
                  <tbody>
                    <tr>
                      <th>User</th>
                      <th>Auth</th>
                      <th>Delete</th>
                    </tr>
                    {Object.entries(authUsers).map(([user, auth]) => (
                      <tr key={user}>
                        <td>{user}</td>
                        <td>{auth}</td>
                        <td>
                          <button
                            className="btn btn p-0"
                            onClick={() => {
                              deleteUserAuth(user);
                            }}
                          >
                            <MdOutlineDeleteOutline />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <span></span>
              )}
            </div>

            <details className="custom-details">
              <summary>Add authorization</summary>
              <select
                id="select-auth"
                className="form-select"
                onChange={(e) => {
                  setCurrentAuth(e.target.value);
                }}
              >
                <option></option>
                <option>manager</option>
                <option>annotator</option>
              </select>
              <button
                onClick={() => {
                  console.log(currentUser);
                  console.log(currentAuth);
                  if (currentUser && currentAuth) {
                    addUserAuth(currentUser, currentAuth);
                  } else
                    notify({
                      type: 'error',
                      message: 'Please select a user, a project and a right',
                    });
                  reFetchUsers();
                }}
                className="btn btn-primary me-2 mt-2"
              >
                Add rights
              </button>
            </details>
          </div>
        </div>
      </div>
    </PageLayout>
  );
};

// list of users associated + possibility to destroy them
// button to add user > menu with name/password/status
// refetch la page
