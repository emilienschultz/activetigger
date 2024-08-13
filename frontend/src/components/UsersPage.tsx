import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { MdOutlineDeleteOutline } from 'react-icons/md';

import {
  useCreateUser,
  useDeleteUser,
  useDeleteUserAuthProject,
  useGetProjectUsers,
  useGetUsers,
  useUserProjects,
} from '../core/api';
import { PageLayout } from './layout/PageLayout';

interface newUser {
  username: string;
  password: string;
  status: string;
}

export const UsersPage: FC = () => {
  const projects = useUserProjects();
  const [currentProjectSlug, setCurrentProjectSlug] = useState('');
  const [refreshComponent, setRefreshComponent] = useState(false);
  const [currentUser, setCurrentUser] = useState('');
  const { authUsers } = useGetProjectUsers(currentProjectSlug);
  const { deleteUserAuth } = useDeleteUserAuthProject(currentProjectSlug);
  const { users } = useGetUsers();
  const { deleteUser } = useDeleteUser();
  const { createUser } = useCreateUser();

  const { handleSubmit, register } = useForm<newUser>();
  const onSubmit: SubmitHandler<newUser> = async (data) => {
    createUser(data.username, data.password, data.status);
    console.log(data);
  };

  return (
    <PageLayout currentPage="users">
      <div className="container-fluid">
        <div className="row">
          <div className="col-6">
            <h2 className="subsection">Manage users and auth</h2>
            <div className="row">
              <h4 className="subsection">Create user</h4>
              <form onSubmit={handleSubmit(onSubmit)}>
                <div className="d-flex align-items-center">
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
                </div>
                <div className="col-3">
                  <select {...register('status')} className="me-2 mt-2">
                    <option>manager</option>
                    <option>annotator</option>
                  </select>
                  <button className="btn btn-primary me-2 mt-2">Create user</button>
                </div>
              </form>
            </div>
            <div className="row">
              <h4 className="subsection">Delete user</h4>
            </div>
            <div className="row">
              <div className="col-4 d-flex align-items-center">
                <select
                  id="select-user"
                  className="form-select"
                  onChange={(e) => {
                    setCurrentUser(e.target.value);
                  }}
                >
                  {users && users.map((e) => <option key={e}>{e}</option>)}
                </select>
                <button
                  className="btn btn p-0"
                  onClick={() => {
                    deleteUser(currentUser);
                  }}
                >
                  <MdOutlineDeleteOutline size={30} />
                </button>
              </div>
            </div>
            <div className="row">
              <h4 className="subsection">Manage project rights</h4>
            </div>
            <div className="row">
              <div className="col-4">
                <label htmlFor="projectSlug">Available projects</label>
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
                    <table>
                      <tr>
                        <th>User</th>
                        <th>Auth</th>
                        <th>Delete</th>
                      </tr>
                      {Object.entries(authUsers).map(([user, auth]) => (
                        <tr>
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
                    </table>
                  ) : (
                    <span>Choose a project</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </PageLayout>
  );
};

// list of users associated + possibility to destroy them
// button to add user > menu with name/password/status
// refetch la page
