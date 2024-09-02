import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { FaPlusCircle, FaRegTrashAlt } from 'react-icons/fa';

import { useAddScheme, useDeleteScheme } from '../core/api';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { SchemeModel } from '../types';

/*
 * Select current scheme
 */

export const SelectCurrentScheme: FC = () => {
  const { notify } = useNotifications();

  // get element from the context
  const {
    appContext: { currentProject, currentScheme },
    setAppContext,
  } = useAppContext();

  const availableSchemes = currentProject ? Object.keys(currentProject.schemes.available) : [];

  // select a default scheme if not
  if (!currentScheme && availableSchemes.length > 0) {
    setAppContext((state) => ({
      ...state,
      currentScheme: availableSchemes[0],
    }));
    notify({ type: 'success', message: `Scheme ${currentScheme} selected by default` });
  }

  // put the current scheme in the context on change
  const handleSelectScheme = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setAppContext((state) => ({
      ...state,
      currentScheme: event.target.value,
    }));
    notify({ type: 'success', message: 'Scheme selected' });
  };

  return (
    <div className="row">
      <div className="d-flex align-items-center mb-3">
        <label htmlFor="scheme-selected">Current scheme</label>
        <select
          id="scheme-selected"
          className="form-select"
          onChange={handleSelectScheme}
          value={currentScheme ? currentScheme : availableSchemes[0]}
        >
          {availableSchemes.map((element) => (
            <option key={element} value={element} selected={element === currentScheme}>
              {element}
            </option>
          ))}{' '}
        </select>
      </div>
    </div>
  );
};

/* Manage schemes
 * Select ; Delete ; Add
 */

export const SchemesManagement: FC<{ projectSlug: string }> = ({ projectSlug }) => {
  // get element from the context
  const {
    appContext: { currentScheme, reFetchCurrentProject },
  } = useAppContext();

  // hooks to use the objets
  const { register, handleSubmit } = useForm<SchemeModel>({});
  const { notify } = useNotifications();

  // hook to get the api call
  const addScheme = useAddScheme(projectSlug);
  const deleteScheme = useDeleteScheme(projectSlug, currentScheme || null);

  // state for displaying the new scheme menu
  const [showCreateNewScheme, setShowCreateNewScheme] = useState(false);
  const handleIconClick = () => {
    setShowCreateNewScheme(!showCreateNewScheme);
  };

  // action to create the new scheme
  const createNewScheme: SubmitHandler<SchemeModel> = async (formData) => {
    try {
      await addScheme(formData.name);
      if (reFetchCurrentProject) reFetchCurrentProject();
      notify({ type: 'success', message: `Scheme ${formData.name} created` });
    } catch (error) {
      notify({ type: 'error', message: error + '' });
    }
    setShowCreateNewScheme(!showCreateNewScheme);
  };

  // action to delete scheme
  const deleteSelectedScheme = async () => {
    //TODO: try catch and throw
    await deleteScheme();
    if (reFetchCurrentProject) reFetchCurrentProject();
  };
  return (
    <div>
      <div className="row">
        <div className="d-flex align-items-center mb-3">
          <SelectCurrentScheme />

          <button onClick={deleteSelectedScheme} className="btn btn p-0 m-1">
            <FaRegTrashAlt size={20} />
          </button>
          <button onClick={handleIconClick} className="btn p-0 m-1">
            <FaPlusCircle size={20} />
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
