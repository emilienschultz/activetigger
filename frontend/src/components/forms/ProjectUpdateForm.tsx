import { FC } from 'react';
import { Controller, SubmitHandler, useForm } from 'react-hook-form';
import Select from 'react-select';

import { useParams } from 'react-router-dom';
import { useUpdateProject } from '../../core/api';
import { useAppContext } from '../../core/context';
import { useNotifications } from '../../core/notifications';
import { ProjectUpdateModel } from '../../types';

export const ProjectUpdateForm: FC = () => {
  const { projectName } = useParams();

  const {
    appContext: { currentProject: project },
  } = useAppContext();
  const columnsSelect =
    project && project.params.all_columns
      ? project.params.all_columns.map((e) => ({ value: e, label: e })) || []
      : [];
  const langages = [
    { value: 'en', label: 'English' },
    { value: 'fr', label: 'French' },
    { value: 'de', label: 'German' },
    { value: 'cn', label: 'Chinese' },
  ];
  const {
    register,
    control,
    handleSubmit,
    formState: { isDirty },
  } = useForm<ProjectUpdateModel>({
    defaultValues: {
      project_name: project ? project.params.project_name : '',
      language: project ? project.params.language : '',
      cols_context: project ? project.params.cols_context : [],
      cols_text: project ? project.params.cols_text : [],
    },
  });
  const { notify } = useNotifications();
  const updateProject = useUpdateProject(projectName || null);

  // action when form validated
  const onSubmit: SubmitHandler<ProjectUpdateModel> = async (formData) => {
    if (!formData.cols_text) {
      notify({ type: 'error', message: 'Please select a text column' });
      return;
    }
    if (!isDirty) {
      notify({ type: 'info', message: 'No changes detected' });
      return;
    }
    updateProject(formData);
  };

  return (
    <div className="container-fluid">
      <form onSubmit={handleSubmit(onSubmit)} className="form-frame">
        <div>
          <label className="form-label" htmlFor="project_name">
            Change name
          </label>
          <input
            className="form-control"
            id="project_name"
            type="text"
            {...register('project_name')}
          />
        </div>

        <div>
          <div>
            <label className="form-label" htmlFor="cols_text">
              Change text columns.
            </label>
            <Controller
              name="cols_text"
              control={control}
              render={({ field: { onChange, value } }) => (
                <Select
                  options={columnsSelect}
                  isMulti
                  value={columnsSelect.filter((option) => value?.includes(option.value))}
                  onChange={(selectedOptions) => {
                    onChange(selectedOptions ? selectedOptions.map((option) => option.value) : []);
                  }}
                />
              )}
            />

            <label className="form-label" htmlFor="language">
              Change language of the corpus (for tokenization and word segmentation)
            </label>
            <select className="form-control" id="language" {...register('language')}>
              {langages.map((lang) => (
                <option key={lang.value} value={lang.value}>
                  {lang.label}
                </option>
              ))}
            </select>

            <label className="form-label" htmlFor="cols_context">
              Change contextual information columns
            </label>

            <Controller
              name="cols_context"
              control={control}
              render={({ field: { onChange, value } }) => (
                <Select
                  options={columnsSelect}
                  isMulti
                  value={columnsSelect.filter((option) => value?.includes(option.value))}
                  onChange={(selectedOptions) => {
                    onChange(selectedOptions ? selectedOptions.map((option) => option.value) : []);
                  }}
                />
              )}
            />
          </div>
          <button type="submit" className="btn btn-primary form-button">
            Modify
          </button>
        </div>
      </form>
    </div>
  );
};
