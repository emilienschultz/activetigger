import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { IoIosAddCircle } from 'react-icons/io';
import { MdOutlineDeleteOutline } from 'react-icons/md';

import { useAddScheme, useDeleteScheme } from '../../core/api';
import { useAppContext } from '../../core/context';
import { useNotifications } from '../../core/notifications';
import { SchemeModel } from '../../types';

interface SchemesManagementProps {
  available_schemes: string[];
  projectSlug: string;
  reFetchProject: () => void;
}

/* Manage schemes
 * Select ; Delete ; Add
 */

export const SchemesManagement: FC<SchemesManagementProps> = ({
  available_schemes,
  projectSlug,
  reFetchProject,
}) => {
  // manage context
  const {
    appContext: { current_scheme },
    setAppContext,
  } = useAppContext();

  // hooks to use the objets
  const { register, handleSubmit } = useForm<SchemeModel>({});
  const { notify } = useNotifications();

  // hook to get the api call
  const addScheme = useAddScheme(projectSlug);
  const deleteScheme = useDeleteScheme(projectSlug, current_scheme);

  // state for displaying the new scheme menu
  const [showCreateNewScheme, setShowCreateNewScheme] = useState(false);
  const handleIconClick = () => {
    setShowCreateNewScheme(!showCreateNewScheme);
  };

  // put the current scheme in the context on change
  const handleSelectScheme = (event: any) => {
    setAppContext((state) => ({
      ...state,
      current_scheme: event.target.value,
    }));
    console.log(`Current scheme ${event.target.value}`);
  };

  // action to create the new scheme
  const createNewScheme: SubmitHandler<SchemeModel> = async (formData) => {
    try {
      addScheme(formData.name);
      notify({ type: 'success', message: `Scheme ${formData.name} created` });
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    reFetchProject();
    setShowCreateNewScheme(!showCreateNewScheme);
  };

  // action to delete scheme
  const deleteSelectedScheme = async () => {
    //TODO: try catch and throw
    await deleteScheme();
    reFetchProject();
  };

  return (
    <div className="container-fluid">
      <div className="row">
        <label className="form-label" htmlFor="scheme-selected">
          Select current scheme
        </label>
        <div>
          <select
            id="scheme-selected"
            className="form-select-lg col-3 mb-3"
            onChange={handleSelectScheme}
          >
            <option></option> {/*empty possibility*/}
            {available_schemes.map((element) => (
              <option key={element} value={element}>
                {element}
              </option>
            ))}{' '}
          </select>
          <button onClick={deleteSelectedScheme} className="btn btn p-0">
            <MdOutlineDeleteOutline size={30} />
          </button>
          <button onClick={handleIconClick} className="btn p-0">
            <IoIosAddCircle size={30} />
          </button>
        </div>
      </div>
      <div>
        {
          // only display if click on the add button
          showCreateNewScheme && (
            <div className="row">
              <form onSubmit={handleSubmit(createNewScheme)}>
                <div className="secondary-panel col-4">
                  <label className="form-label" htmlFor="scheme_name">
                    New scheme
                  </label>
                  <input
                    className="form-control"
                    id="scheme_name"
                    type="text"
                    {...register('name')}
                  />
                  <button className="btn btn-primary btn-validation">Create</button>
                </div>
              </form>
            </div>
          )
        }
      </div>
    </div>
  );
};
